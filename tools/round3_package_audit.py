"""Read-only independent TapChain package audit; never opens GUI/controller."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = APP_ROOT.parent/'.work/round3-implementation-20261003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    if args.output.exists():
        raise FileExistsError('Preserve prior package audit')
    sys.path.insert(0,str(APP_ROOT/'tools'))
    from round3_replay_compare import source_fingerprint
    sys.path.insert(0,str(package))
    from agent.music.build_identity import verify_manifest,current_identity
    from agent.common import data_root
    checks, details = {}, {}
    manifest = json.loads((package/'build-manifest.json').read_text(encoding='utf-8'))
    verify_manifest(package,manifest)
    checks['all_manifest_source_and_dependency_files_verified'] = True
    details['manifest'] = {'build_id':manifest['build_id'],'source_files':len(manifest['files']),
        'dependency_files':len(manifest.get('dependency_files',{})),
        'current_identity_read_only':current_identity()}
    details['source_fingerprint'] = source_fingerprint(package)
    checks['source_matches_final_replay'] = (details['source_fingerprint']['sha256']==
        '67367e2464d13b0d26a93e821ec460dc2a62ebf5635effe7488a0f27f5ad2a82')
    checks['candidate_marker'] = (package/'candidate-package.marker').is_file()
    # No directory creation: data_root's only write is mocked in this process.
    with patch.dict(os.environ,{'MAES_DATA_DIR':''}),patch.object(Path,'mkdir'):
        default_data = data_root().resolve()
    checks['default_data_is_package_local'] = default_data==(package/'user-data').resolve()
    details['default_data_root'] = str(default_data)
    details['isolation_warning'] = 'Explicit MAES_DATA_DIR has priority over candidate marker. Clear it or set it to this package/user-data when launching; default isolation is verified without creating any file.'
    old = WORK_ROOT/'baseline/candidate-state/user-data'
    state_checks = {}
    for name in ('calibration/music.json','music_touch.json'):
        left,right = old/name,package/'user-data'/name
        state_checks[name] = {'source_exists':left.is_file(),'package_exists':right.is_file(),
            'source_sha256':sha(left) if left.is_file() else None,
            'package_sha256':sha(right) if right.is_file() else None}
        state_checks[name]['matches'] = left.is_file() and right.is_file() and sha(left)==sha(right)
    checks['calibration_and_touch_equal_old_markerfix'] = all(x['matches'] for x in state_checks.values())
    details['state_checks'] = state_checks
    forbidden = [str(path.relative_to(package)) for path in package.rglob('*')
        if (path.is_dir() and path.name in {'logs','instances','debug','temp','backup','__pycache__'})
        or path.name in {'music_last_result.json','music_preflight.json'}
        or path.suffix.lower() in {'.pyc','.pyo'}]
    checks['no_old_transient_state'] = not forbidden
    details['forbidden_paths'] = forbidden
    config = json.loads((package/'config/config.json').read_text(encoding='utf-8'))
    checks['preview_flag_true'] = config.get('UI.LiveView.EnableLiveView') is True
    details['preview_config'] = config
    settings = json.loads((package/'appsettings.json').read_text(encoding='utf-8'))
    checks['no_auto_start'] = str(settings.get('NoAutoStart')).lower()=='true'
    details['appsettings'] = settings
    source_interface = json.loads((APP_ROOT/'interface.json').read_text(encoding='utf-8'))
    package_interface = json.loads((package/'interface.json').read_text(encoding='utf-8'))
    baseline_interface = json.loads((WORK_ROOT/'baseline/source/interface.json').read_text(encoding='utf-8'))
    checks['interface_equals_validated_source'] = package_interface==source_interface
    normalize = lambda value:{key:item for key,item in value.items() if key not in {'title','version'}}
    checks['interface_behavior_equals_baseline'] = normalize(package_interface)==normalize(baseline_interface)
    pipelines = {}
    # Resource roots are resource/base/pipeline, not resource/pipeline.
    # Include default_pipeline and GUI layout JSON as additional protection.
    for path in sorted((package/'resource').rglob('*.json')):
        relative = path.relative_to(package)
        baseline = WORK_ROOT/'baseline/source'/relative
        source = APP_ROOT/relative
        pipelines[str(relative)] = {'package_sha256':sha(path),'source_matches':source.is_file() and sha(source)==sha(path),
                                   'baseline_matches':baseline.is_file() and sha(baseline)==sha(path)}
    checks['all_pipelines_equal_source_and_frozen_baseline'] = bool(pipelines) and all(
        x['source_matches'] and x['baseline_matches'] for x in pipelines.values())
    details['pipelines'] = pipelines
    archive_key_files = {}
    with zipfile.ZipFile(args.archive) as archive:
        for name in ('build-manifest.json','interface.json','config/config.json','appsettings.json',
                     'candidate-package.marker','user-data/calibration/music.json','user-data/music_touch.json'):
            try:
                archive_key_files[name] = hashlib.sha256(archive.read(name)).hexdigest()==sha(package/name)
            except KeyError:
                archive_key_files[name] = False
    checks['archive_critical_files_equal_tree'] = all(archive_key_files.values())
    details['archive_key_files'] = archive_key_files
    report = {'schema':1,'mode':'read-only-independent-package-audit','package':str(package),
        'archive':str(args.archive.resolve()),'checks':checks,'passed':all(checks.values()),'details':details,
        'limitations':'No GUI/controller/simulator launched. All package reads only; Path.mkdir mocked for default data-root query. Manifest covers source/resource/dependencies; mutable config/calibration separately audited. Preview true is static configuration evidence, not proof of GUI rendering. Historical package preservation is covered by separate full protection scan. No BM/FC inference.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':checks,'build_id':manifest['build_id']},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

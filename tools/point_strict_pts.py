"""Strict same-pixels/PTS point qualification benchmark, no live controller.

The complete reviewed round3_strict_pts implementation is reused in memory;
named, count-checked substitutions only supply explicit fixtures/output paths
and put both dispatches after this frame's visual update. The original tool
is not edited. Both tools and helper/source hashes guard every comparison.

Decode one complete clip, wait for FFmpeg to exit, then time qualification.
This is a fixed-PTS CPU/wall fixture, NOT a production runtime-loop replay,
gesture timing validation or Bad/Miss inference. Use loop_replay separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve()
TEMPLATE = HERE.with_name('round3_strict_pts.py')


def substitute(source, old, new, *, count=1):
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(f'Reviewed benchmark template changed: expected {count} matches, found {actual}: {old[:96]}')
    return source.replace(old, new)


def prepare(work_root, inventory):
    # Read the complete implementation, not selected fragments or a summary.
    source = TEMPLATE.read_text(encoding='utf-8')
    source = substitute(source, "WORK_ROOT = APP_ROOT.parent/'.work/round3-implementation-20261003'",
                        f'WORK_ROOT = Path({str(work_root)!r})\nINVENTORY_PATH = Path({str(inventory)!r})')
    source = substitute(source, "'version':2,'source_clock':", "'version':3,'source_clock':")
    source = substitute(source, "'dispatch':'both nonwaiting _execute_due calls, including instant mock transport overhead',",
                        "'dispatch':'both nonwaiting _execute_due calls AFTER same-frame observations update; instant mock transport',\n"
                        "    'order':'external gold prepass, provider/mask, engine.update, old-pending dispatch, planning, final dispatch',")
    source = substitute(source, "output = APP_ROOT/'temp/round3-implementation'/args.label",
                        "output = WORK_ROOT/'strict-source-pts'/args.label")
    source = substitute(source, "inventory = json.loads((WORK_ROOT/'clip-fixtures.json').read_text())",
                        "inventory = json.loads(INVENTORY_PATH.read_text(encoding='utf-8'))\n"
                        "    inventory_hash = hashlib.sha256(INVENTORY_PATH.read_bytes()).hexdigest()")
    tool_tuple = "(Path(__file__),APP_ROOT/'tools/tap_replay.py',APP_ROOT/'tools/workspace_paths.py')"
    source = substitute(source, tool_tuple,
                        "(Path(__file__),TEMPLATE_PATH,APP_ROOT/'tools/tap_replay.py',APP_ROOT/'tools/workspace_paths.py')", count=2)
    old_inputs = """        package = WORK_ROOT/'baseline/candidate-state'
        calibration_path = package/'user-data/calibration/music.json'
        config_path = package/f'logs/tap-traces/{item["run"]}.jsonl'
        calibration = MusicCalibrationData(**json.loads(calibration_path.read_text())['profiles']['7@1280x720'])"""
    new_inputs = """        calibration_path = Path(item['calibration'])
        config_path = Path(item['config'])
        fixture_file_hashes = {'config':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                               'calibration':hashlib.sha256(calibration_path.read_bytes()).hexdigest()}
        calibration = MusicCalibrationData(**json.loads(calibration_path.read_text(encoding='utf-8'))['profiles']['7@1280x720'])"""
    source = substitute(source, old_inputs, new_inputs)
    source = substitute(source,
        "if (identity['config_hash']!='6200cae263aa15a7'\n                or identity['effective_calibration_hash']!='20bfdc72be93e1b61d5d2d8803ca39562ac72d77b55d9c6b0be31a6dd3371714'):",
        "if (identity['config_hash']!=item['config_hash']\n                or identity['effective_calibration_hash']!=item['effective_calibration_hash']):")
    source = substitute(source,
        "options = SimpleNamespace(video=APP_ROOT.parent/f'test-materials/music/20261003test3-{item[\"video_index\"]}.mp4',",
        "options = SimpleNamespace(video=Path(item['video']),")
    # Eliminate the old transport's compatibility fallback: both compared
    # sources must expose complete per-flick receipt contracts or fail clearly.
    source = substitute(source, "import agent.music.executor as bindings",
        "import agent.music.executor as bindings\n"
        "    if not all(hasattr(bindings, name) for name in ('FlickInputReceipt', 'FlickSubmission')):\n"
        "        raise RuntimeError('Strict benchmark requires complete flick receipt API; no fallback')")
    dispatch = """            begin = time.perf_counter()
            begin_cpu = time.process_time()
            runtime._execute_due(executor,pending,pts,metrics,engine,wait=False)
            before_dispatch_ms = (time.perf_counter()-begin)*1000.
            before_dispatch_process_ms = (time.process_time()-begin_cpu)*1000.
"""
    source = substitute(source, dispatch, '')
    update_end = """            update_ms = (time.perf_counter()-begin)*1000.
            update_process_ms = (time.process_time()-begin_cpu)*1000.
"""
    source = substitute(source, update_end, update_end+dispatch)
    source = substitute(source, "report = {'schema':2,'mode':'strict-source-PTS-qualification-CPU','clip':item,",
        "report = {'schema':3,'mode':'point-strict-source-PTS-qualification-wall','clip':item,\n"
        "            'inventory_hash':inventory_hash,\n"
        "            'inventory_changed':hashlib.sha256(INVENTORY_PATH.read_bytes()).hexdigest()!=inventory_hash,\n"
        "            'fixture_file_hashes':fixture_file_hashes,\n"
        "            'fixture_files_changed':fixture_file_hashes!={'config':hashlib.sha256(config_path.read_bytes()).hexdigest(),\n"
        "                                                        'calibration':hashlib.sha256(calibration_path.read_bytes()).hexdigest()},")
    source = substitute(source,
        "old = json.loads((APP_ROOT/'temp/round3-implementation'/args.compare_label/f'{name}.json').read_text())",
        "old = json.loads((WORK_ROOT/'strict-source-pts'/args.compare_label/f'{name}.json').read_text(encoding='utf-8'))")
    source = substitute(source, "guards = {\n                'same_pixels_and_pts':",
        "guards = {\n                'same_inventory':report['inventory_hash']==old.get('inventory_hash'),\n"
        "                'inventory_stable':not report['inventory_changed'] and not old.get('inventory_changed',True),\n"
        "                'same_fixture_files':report['fixture_file_hashes']==old.get('fixture_file_hashes'),\n"
        "                'fixture_files_stable':not report['fixture_files_changed'] and not old.get('fixture_files_changed',True),\n"
        "                'same_pixels_and_pts':")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--branch-root',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--clips',nargs='*')
    parser.add_argument('--compare-label')
    parser.add_argument('--prepare-only',action='store_true',help='Validate reviewed tool substitutions and memory estimates; do not decode or measure')
    args = parser.parse_args()
    work_root, inventory = args.work_root.resolve(), args.inventory.resolve()
    if not inventory.is_file() or not (args.branch_root/'agent').is_dir():
        parser.error('Explicit inventory and branch agent source must exist')
    if not args.label.replace('-','').replace('_','').isalnum():
        parser.error('Use letters, digits, hyphens or underscores for labels')
    items = json.loads(inventory.read_text(encoding='utf-8'))
    if not isinstance(items,list) or not items:
        parser.error('Inventory must be a nonempty list')
    required = {'name','video','config','calibration','start','end','config_hash','effective_calibration_hash'}
    for item in items:
        if not required <= item.keys() or not 0 <= item['start'] < item['end']:
            parser.error('Invalid explicit fixture fields or time interval')
        for key in ('video','config','calibration'):
            if not Path(item[key]).is_absolute() or not Path(item[key]).is_file():
                parser.error(f'Fixture {key} must be an existing absolute path: {item.get("name")}')
    if len({item['name'] for item in items}) != len(items):
        parser.error('Duplicate fixture name')
    if args.clips and not set(args.clips) <= {item['name'] for item in items}:
        parser.error('Unknown clip')
    source = prepare(work_root,inventory)
    code = compile(source,str(HERE),'exec')
    estimates = {item['name']:math.ceil((item['end']-item['start'])*31)*1280*720*3
                 for item in items if not args.clips or item['name'] in args.clips}
    if args.prepare_only:
        print(json.dumps({'ready':True,'fixtures':len(estimates),'max_predecode_estimated_bytes':max(estimates.values()),
                          'required_free_bytes_including_2gib_reserve':max(estimates.values())+2*1024**3,
                          'estimates_by_clip':estimates,'template_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
                          'measurement':'qualification perf_counter wall P95; every decoded PTS once; not a real runtime loop',
                          'dispatch_order':'same-frame observations before both nonwaiting dispatch calls',
                          'decoded_or_measured':False},indent=2))
        return 0
    old_argv = sys.argv
    sys.argv = [str(HERE),'--branch-root',str(args.branch_root.resolve()),'--label',args.label]
    if args.clips:
        sys.argv.extend(['--clips',*args.clips])
    if args.compare_label:
        sys.argv.extend(['--compare-label',args.compare_label])
    try:
        namespace = {'__file__':str(HERE),'__name__':'point_strict_pts_implementation','TEMPLATE_PATH':TEMPLATE}
        exec(code,namespace)
        return namespace['main']()
    finally:
        sys.argv = old_argv


if __name__=='__main__':
    raise SystemExit(main())

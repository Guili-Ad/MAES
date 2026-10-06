"""Create/verify source manifests, package trees and archived package contents."""
import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.music.build_identity import create_manifest, verify_manifest, seal_package


def verify_archive(path):
    with tempfile.TemporaryDirectory(prefix='maes-package-check-') as temporary:
        # Windows TEMP may contain an 8.3 alias (ADMINI~1). Compare canonical
        # paths on both sides; do not relax the archive traversal guard.
        root = Path(temporary).resolve()
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                target = (root / item.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError('Unsafe archive path')
            archive.extractall(root)
        manifest = json.loads((root / 'build-manifest.json').read_text(encoding='utf-8'))
        verify_manifest(root, manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, help='Validated source whose files were copied into --root')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.archive:
        manifest = verify_archive(args.archive)
    elif args.verify:
        manifest = json.loads((args.root / 'build-manifest.json').read_text(encoding='utf-8'))
        verify_manifest(args.root, manifest)
    else:
        manifest = create_manifest(args.source_root or args.root)
        verify_manifest(args.root, manifest)
        if args.source_root:
            manifest = seal_package(args.root, manifest)
            verify_manifest(args.root, manifest)
        (args.root / 'build-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'build_id': manifest['build_id'], 'verified_files': len(manifest['files']),
                      'verified_dependencies': len(manifest.get('dependency_files', {}))}))


if __name__ == '__main__':
    main()

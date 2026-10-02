"""Content identity shared by source runs, packages and replay reports."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

VERSION = "v1.0.1-opt-candidate"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def create_manifest(root: Path) -> dict:
    paths = [root / "interface.json", root / "requirements.lock", root / "runtime/manifest.json"]
    paths += sorted((root / "agent").rglob("*.py"))
    paths += sorted(path for path in (root / "resource").rglob("*") if path.is_file())
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths if path.is_file()}
    dependencies = {}
    for name in ("runtime/manifest.json", "vendor/manifest.json", "vendor/maaframework/manifest.json"):
        path = root / name
        if path.is_file():
            dependencies[name] = json.loads(path.read_text(encoding="utf-8"))
    body = {"version": VERSION, "files": files, "dependencies": dependencies}
    return {"schema": 1, "build_id": f"{VERSION}-{digest(body)[:12]}", **body}


def seal_package(root: Path, manifest: dict) -> dict:
    """Hash the actual shipped dependencies once, outside the live loop."""
    paths = set((root / 'runtime').rglob('*'))
    paths.update((root / 'runtimes').rglob('*'))
    paths.update(path for path in root.iterdir() if path.suffix.lower() in ('.exe', '.dll'))
    dependencies = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(paths) if path.is_file()
                    and path.suffix.lower() not in ('.pyc', '.pyo', '.whl')}
    body = {key: manifest[key] for key in ('version', 'files', 'dependencies')}
    body['dependency_files'] = dependencies
    return {'schema': 2, 'build_id': f"{body['version']}-{digest(body)[:12]}", **body}


def verify_manifest(root: Path, manifest: dict, *, dependencies: bool = True) -> None:
    body = {key: manifest[key] for key in ("version", "files", "dependencies")}
    if 'dependency_files' in manifest:
        body['dependency_files'] = manifest['dependency_files']
    if manifest.get("build_id") != f"{manifest['version']}-{digest(body)[:12]}":
        raise ValueError("Build manifest identity mismatch")
    files = dict(manifest['files'])
    if dependencies:
        files.update(manifest.get('dependency_files', {}))
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Build file missing or outside package: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Build file hash mismatch: {name}")


@lru_cache(maxsize=1)
def current_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / "build-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else create_manifest(root)
    # Dependency bytes were checked while building/unpacking. Do not scan the
    # entire embedded runtime at agent startup; still verify live source files.
    verify_manifest(root, manifest, dependencies=False)
    return {"version": manifest["version"], "build_id": manifest["build_id"]}

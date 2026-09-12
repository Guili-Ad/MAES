from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


APP_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = APP_ROOT / "resource" / "base" / "pipeline"
IMAGE_ROOT = APP_ROOT / "resource" / "base" / "image"


def iter_values(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_values(child)


def node_name_from_reference(reference: str) -> str:
    if reference.startswith("[") and "]" in reference:
        return reference.split("]", 1)[1]
    return reference


class ProjectCheck:
    def __init__(self, release: bool, require_runtime: bool) -> None:
        self.release = release
        self.require_runtime = require_runtime
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def load_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.error(f"Invalid JSON {path.relative_to(APP_ROOT)}: {error}")
            return None

    def load_nodes(self) -> dict[str, tuple[Path, dict[str, Any]]]:
        nodes: dict[str, tuple[Path, dict[str, Any]]] = {}
        for path in sorted(PIPELINE_ROOT.rglob("*.json")):
            data = self.load_json(path)
            if not isinstance(data, dict):
                continue
            for node_name, node in data.items():
                if node_name in nodes:
                    self.error(
                        f"Duplicate node {node_name}: {path.relative_to(APP_ROOT)} and "
                        f"{nodes[node_name][0].relative_to(APP_ROOT)}"
                    )
                    continue
                if not isinstance(node, dict):
                    self.error(f"Node {node_name} is not an object")
                    continue
                nodes[node_name] = (path, node)
        return nodes

    def check_pipeline_shape(self, nodes: dict[str, tuple[Path, dict[str, Any]]]) -> None:
        references: list[tuple[str, str]] = []
        custom_actions: set[str] = set()
        templates: list[tuple[str, str]] = []
        for node_name, (_path, node) in nodes.items():
            if "interrupt" in node:
                self.error(f"Deprecated interrupt field remains in node {node_name}")
            recognition = node.get("recognition")
            action = node.get("action")
            if not isinstance(recognition, dict) or not isinstance(recognition.get("type"), str):
                self.error(f"Node {node_name} does not use Pipeline V2 recognition format")
            if not isinstance(action, dict) or not isinstance(action.get("type"), str):
                self.error(f"Node {node_name} does not use Pipeline V2 action format")

            for field in ("next", "on_error"):
                value = node.get(field, [])
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list):
                    self.error(f"Node {node_name}.{field} must be an array")
                    continue
                for target in value:
                    if isinstance(target, str):
                        references.append((node_name, node_name_from_reference(target)))

            if isinstance(recognition, dict):
                params = recognition.get("param", {})
                if recognition.get("type") in {"And", "Or"} and isinstance(params, dict):
                    for field in ("all_of", "any_of"):
                        for target in params.get(field, []) or []:
                            if isinstance(target, str):
                                references.append((node_name, node_name_from_reference(target)))
                template_value = params.get("template") if isinstance(params, dict) else None
                if isinstance(template_value, str):
                    templates.append((node_name, template_value))
                elif isinstance(template_value, list):
                    templates.extend(
                        (node_name, item) for item in template_value if isinstance(item, str)
                    )

            if isinstance(action, dict) and action.get("type") == "Custom":
                params = action.get("param", {})
                custom_action = params.get("custom_action") if isinstance(params, dict) else None
                if isinstance(custom_action, str):
                    custom_actions.add(custom_action)
                else:
                    self.error(f"Custom action node {node_name} has no custom_action identifier")

        for source, target in references:
            if target and target not in nodes:
                self.error(f"Node {source} references missing node {target}")
        for source, template in templates:
            if "{" in template:
                continue
            if not (IMAGE_ROOT / template).is_file():
                self.error(f"Node {source} references missing template {template}")

        registered_actions = self.registered_actions()
        for action_name in sorted(custom_actions - registered_actions):
            self.error(f"Pipeline uses unregistered custom action {action_name}")

    def registered_actions(self) -> set[str]:
        pattern = re.compile(r"AgentServer\.custom_action\(\s*[\"']([^\"']+)[\"']\s*\)")
        actions: set[str] = set()
        for path in (APP_ROOT / "agent").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            actions.update(pattern.findall(text))
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as error:
                self.error(f"Python syntax error in {path.relative_to(APP_ROOT)}: {error}")
        return actions

    def check_interface(self, nodes: dict[str, tuple[Path, dict[str, Any]]]) -> None:
        interface = self.load_json(APP_ROOT / "interface.json")
        if not isinstance(interface, dict):
            return
        if interface.get("interface_version") != 2:
            self.error("interface.json must declare interface_version: 2")
        agent = interface.get("agent", {})
        if agent.get("child_exec") != "runtime/python/python.exe":
            self.error("Agent must use the bundled offline Python runtime")
        child_args = agent.get("child_args", [])
        if not isinstance(child_args, list) or "-B" not in child_args:
            self.error("Agent must disable Python bytecode generation with -B")
        controllers = interface.get("controller", [])
        if len(controllers) != 1 or controllers[0].get("type") != "Adb":
            self.error("The first release must expose exactly one generic Adb controller")
        for task in interface.get("task", []):
            entry = task.get("entry")
            if entry not in nodes:
                self.error(f"Task {task.get('name')} references missing entry {entry}")
            for option_name in task.get("option", []):
                if option_name not in interface.get("option", {}):
                    self.error(f"Task {task.get('name')} references undefined option {option_name}")
            self.check_override_targets(
                task.get("pipeline_override", {}),
                nodes,
                f"task {task.get('name')}",
            )
        for option_name, option in interface.get("option", {}).items():
            option_type = option.get("type")
            if option_type not in {"select", "switch", "checkbox", "input"}:
                self.error(f"Option {option_name} has invalid type {option_type}")
                continue
            if option_type == "input":
                self.check_override_targets(
                    option.get("pipeline_override", {}),
                    nodes,
                    f"option {option_name}",
                )
                continue
            cases = option.get("cases", [])
            case_names = {case.get("name") for case in cases}
            if option.get("default_case") not in case_names:
                self.error(f"Option {option_name} default_case does not match any case")
            if option_type == "switch" and case_names != {"Yes", "No"}:
                self.error(f"Switch option {option_name} must use exactly Yes/No cases")
            for case in cases:
                self.check_override_targets(
                    case.get("pipeline_override", {}),
                    nodes,
                    f"option {option_name}/{case.get('name')}",
                )

    def check_override_targets(
        self,
        override: Any,
        nodes: dict[str, tuple[Path, dict[str, Any]]],
        source: str,
    ) -> None:
        if not isinstance(override, dict):
            self.error(f"{source} pipeline_override must be an object")
            return
        for node_name, patch in override.items():
            if node_name not in nodes:
                self.error(f"{source} overrides undefined node {node_name}")
            if isinstance(patch, dict) and "interrupt" in patch:
                self.error(f"{source} contains deprecated interrupt override")

    def check_portability(self) -> None:
        forbidden_patterns = {
            r"(?i)(?<![A-Za-z0-9_])MAAStudyModel(?![A-Za-z0-9_])": "reference model name/path",
            r"(?i)(?<![A-Za-z0-9_])MMleo(?![A-Za-z0-9_])": "reference project name/path",
            r"(?i)(?<![A-Z])[A-Z]:[\\/]": "absolute Windows path",
            r"127\.0\.0\.1:\d+": "fixed ADB address",
            r"(?i)MuMuPlayer|LDPlayer|Nox": "emulator-specific runtime value",
        }
        scan_roots = [
            APP_ROOT / "agent",
            APP_ROOT / "resource",
            APP_ROOT / "tools",
            APP_ROOT / "packaging",
        ]
        scan_files = [APP_ROOT / "interface.json", APP_ROOT / "README.md"]
        for root in scan_roots:
            scan_files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".py", ".json", ".ps1", ".md", ".txt"}
            )
        for path in scan_files:
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern, label in forbidden_patterns.items():
                if re.search(pattern, text):
                    self.error(f"{path.relative_to(APP_ROOT)} contains forbidden {label}")

    def check_provenance(self) -> None:
        provenance = self.load_json(APP_ROOT / "provenance" / "assets.json")
        if not isinstance(provenance, dict):
            return
        records = provenance.get("assets", [])
        if not isinstance(records, list):
            self.error("provenance/assets.json assets must be an array")
            return

        records_by_path: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                self.error("Asset provenance contains an invalid record")
                continue
            relative = record["path"]
            if relative in records_by_path:
                self.error(f"Duplicate provenance record: {relative}")
                continue
            records_by_path[relative] = record
            if not isinstance(record.get("source"), str) or not record["source"].strip():
                self.error(f"Asset provenance has no source: {relative}")

            asset_path = APP_ROOT / relative
            try:
                asset_path.resolve().relative_to(APP_ROOT.resolve())
            except ValueError:
                self.error(f"Asset provenance escapes the project root: {relative}")
                continue
            if not asset_path.is_file():
                self.error(f"Asset provenance references a missing file: {relative}")
                continue
            digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
            if digest.lower() != str(record.get("sha256", "")).lower():
                self.error(f"Asset provenance hash mismatch: {relative}")

        tracked_files = {
            path.relative_to(APP_ROOT).as_posix()
            for root in (APP_ROOT / "resource" / "base" / "image", APP_ROOT / "resource" / "base" / "model")
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file()
        }
        order_library = APP_ROOT / "agent" / "order_library.json"
        if order_library.is_file():
            tracked_files.add(order_library.relative_to(APP_ROOT).as_posix())
        for relative in sorted(tracked_files - records_by_path.keys()):
            self.error(f"Asset has no provenance record: {relative}")
        for relative in sorted(records_by_path.keys() - tracked_files):
            self.error(f"Provenance record is stale: {relative}")

        blocked = [
            asset
            for asset in records
            if isinstance(asset, dict)
            if not asset.get("release_allowed", False)
        ]
        if blocked:
            message = f"{len(blocked)} prototype/upstream-review resources are not release-approved"
            if self.release:
                self.error(message)
            else:
                self.warn(message)
        if self.release:
            reference_sources = [
                asset
                for asset in records
                if isinstance(asset, dict)
                and str(asset.get("source", "")).lower().startswith("mmleo/")
            ]
            if reference_sources:
                self.error(
                    f"{len(reference_sources)} release resources still identify the reference project as their source"
                )

    def check_transient_artifacts(self) -> None:
        validation_root = APP_ROOT / ".validation"
        if validation_root.exists():
            self.error("Temporary .validation directory must be removed")

        for root in (
            APP_ROOT / "agent",
            APP_ROOT / "tests",
            APP_ROOT / "tools",
            APP_ROOT / "runtime" / "python",
        ):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_dir() and path.name == "__pycache__":
                    self.error(f"Python cache directory remains: {path.relative_to(APP_ROOT)}")
                elif path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}:
                    self.error(f"Python bytecode remains: {path.relative_to(APP_ROOT)}")

        package_defaults = self.load_json(APP_ROOT / "packaging" / "appsettings.json")
        if not isinstance(package_defaults, dict) or package_defaults.get("NoAutoStart") != "True":
            self.error("Packaged MFAAvalonia defaults must disable first-run task auto-start")
        update_defaults = self.load_json(APP_ROOT / "packaging" / "config.json")
        expected_update_defaults = {
            "EnableCheckVersion": False,
            "EnableAutoUpdateResource": False,
            "EnableAutoUpdateMFA": False,
        }
        if update_defaults != expected_update_defaults:
            self.error("Offline package defaults must disable unsupported online update checks")

    def check_runtime(self) -> None:
        runtime_python = APP_ROOT / "runtime" / "python" / "python.exe"
        manifest = APP_ROOT / "runtime" / "manifest.json"
        runtime_data = self.load_json(manifest) if manifest.is_file() else None
        if runtime_data is None:
            self.warn("runtime/manifest.json is missing")
        if self.require_runtime and not runtime_python.is_file():
            self.error("Bundled Python runtime is required but runtime/python/python.exe is missing")
        if runtime_python.is_file():
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    str(runtime_python),
                    "-B",
                    "-c",
                    (
                        "import importlib.metadata,json,struct,sys,numpy;"
                        "print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),"
                        "'bits':struct.calcsize('P')*8,'maafw':importlib.metadata.version('maafw'),"
                        "'numpy':numpy.__version__}))"
                    ),
                ],
                cwd=APP_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
            if result.returncode != 0:
                self.error(f"Bundled Python cannot import locked dependencies: {result.stderr.strip()}")
            else:
                try:
                    versions = json.loads(result.stdout.strip())
                except json.JSONDecodeError as error:
                    self.error(f"Bundled Python version probe returned invalid output: {error}")
                else:
                    expected = {
                        "python": "3.12.10",
                        "bits": 64,
                        "maafw": "5.12.2",
                        "numpy": "2.2.6",
                    }
                    if versions != expected:
                        self.error(f"Bundled Python runtime does not match the lock: {versions}")

        if isinstance(runtime_data, dict):
            wheel_root = APP_ROOT / "runtime" / "wheels"
            for package in runtime_data.get("packages", []):
                wheel = wheel_root / package.get("file", "")
                if not wheel.is_file():
                    self.error(f"Locked wheel is missing: {wheel.name}")
                    continue
                digest = hashlib.sha256(wheel.read_bytes()).hexdigest().upper()
                if digest != str(package.get("sha256", "")).upper():
                    self.error(f"Locked wheel hash mismatch: {wheel.name}")

        for path in (APP_ROOT / "agent").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bpip\s+install\b|subprocess\..*pip", text, re.IGNORECASE):
                self.error(f"Agent performs a runtime package installation: {path.relative_to(APP_ROOT)}")

        vendor_manifest = APP_ROOT / "vendor" / "manifest.json"
        vendor_data = self.load_json(vendor_manifest) if vendor_manifest.is_file() else None
        gui_exe = APP_ROOT / "vendor" / "mfaavalonia" / "MFAAvalonia.exe"
        if not gui_exe.is_file():
            self.error("MFAAvalonia Windows x64 host is missing")
        if isinstance(vendor_data, dict):
            gui = vendor_data.get("gui", {})
            archive = APP_ROOT / "vendor" / gui.get("archive", "")
            if not archive.is_file():
                self.error("MFAAvalonia source archive is missing")
            else:
                digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
                if digest != str(gui.get("archive_sha256", "")).upper():
                    self.error("MFAAvalonia archive hash mismatch")

    def check_framework_load(self) -> None:
        framework_bin = APP_ROOT / "vendor" / "maaframework"
        dll_path = framework_bin / "MaaFramework.dll"
        if not dll_path.is_file():
            self.warn(
                "vendor/maaframework/MaaFramework.dll is unavailable; run tools/bootstrap.ps1; "
                "native bundle load was skipped"
            )
            return
        if not (framework_bin / "LICENSE.md").is_file():
            self.error("vendor/maaframework/LICENSE.md is missing")
        (framework_bin / "plugins").mkdir(parents=True, exist_ok=True)
        self.check_framework_hashes(framework_bin)
        try:
            dll_directory = None
            if hasattr(os, "add_dll_directory"):
                dll_directory = os.add_dll_directory(str(framework_bin))
            library = ctypes.CDLL(str(dll_path))
            library.MaaVersion.restype = ctypes.c_char_p
            version = library.MaaVersion().decode("utf-8")
            if version != "v5.12.2":
                self.error(f"Expected MaaFramework v5.12.2 but loaded {version}")

            library.MaaResourceCreate.restype = ctypes.c_void_p
            library.MaaResourceDestroy.argtypes = [ctypes.c_void_p]
            library.MaaResourcePostBundle.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            library.MaaResourcePostBundle.restype = ctypes.c_int64
            library.MaaResourceWait.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            library.MaaResourceWait.restype = ctypes.c_int32
            resource = library.MaaResourceCreate()
            if not resource:
                self.error("MaaResourceCreate returned null")
                return
            try:
                for bundle in (APP_ROOT / "resource" / "base", APP_ROOT / "resource" / "bside"):
                    if not bundle.is_dir():
                        continue
                    job = library.MaaResourcePostBundle(resource, str(bundle).encode("utf-8"))
                    status = library.MaaResourceWait(resource, job)
                    if status != 3000:
                        self.error(f"MaaFramework failed to load {bundle.relative_to(APP_ROOT)}: status={status}")
            finally:
                library.MaaResourceDestroy(resource)
                if dll_directory is not None:
                    dll_directory.close()
        except (OSError, AttributeError, UnicodeError) as error:
            self.error(f"MaaFramework native validation failed: {error}")

    def check_framework_hashes(self, framework_bin: Path) -> None:
        manifest_path = framework_bin / "manifest.json"
        if not manifest_path.is_file():
            self.warn("vendor/maaframework/manifest.json is missing; native file hashes were not verified")
            return
        data = self.load_json(manifest_path)
        if not isinstance(data, dict):
            self.error("vendor/maaframework/manifest.json is invalid")
            return
        for record in data.get("files", []) or []:
            if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                self.error("vendor/maaframework/manifest.json contains an invalid record")
                continue
            name = record["file"]
            path = framework_bin / name
            if not path.is_file():
                self.error(f"Vendor framework file is missing: {name}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            if digest != str(record.get("sha256", "")).upper():
                self.error(f"Vendor framework hash mismatch: {name}")

    def run(self) -> int:
        nodes = self.load_nodes()
        self.check_pipeline_shape(nodes)
        self.check_interface(nodes)
        self.check_portability()
        self.check_provenance()
        self.check_transient_artifacts()
        self.check_runtime()
        self.check_framework_load()
        for warning in self.warnings:
            print(f"WARNING: {warning}")
        for error in self.errors:
            print(f"ERROR: {error}")
        print(
            f"Checked {len(nodes)} nodes, {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings"
        )
        return 1 if self.errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the independent MAES project")
    parser.add_argument("--release", action="store_true", help="fail on unapproved prototype resources")
    parser.add_argument("--require-runtime", action="store_true", help="require bundled Python")
    args = parser.parse_args()
    return ProjectCheck(args.release, args.require_runtime).run()


if __name__ == "__main__":
    raise SystemExit(main())

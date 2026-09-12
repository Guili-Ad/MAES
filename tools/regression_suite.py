"""v0.1 regression suite: tests + hold contract + synthetic replay + benchmarks.

Behavior-preserving changes must keep the contract/replay sections identical.
Run with the branch-local embedded Python:

  runtime\\python\\python.exe -B tools\\regression_suite.py --output temp/optimization-v0.1/baseline

Then compare after changes:

  runtime\\python\\python.exe -B tools\\regression_suite.py --output temp/optimization-v0.1/after \\
      --baseline temp/optimization-v0.1/baseline

Development-only. Never opens a controller.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "runtime" / "python" / "python.exe"
SYNTHETIC = ROOT / "temp" / "validation" / "tap-chord-v1" / "synthetic.jsonl"
SYNTHETIC_ANNOTATIONS = ROOT / "temp" / "validation" / "tap-chord-v1" / "synthetic.annotations.json"

BEHAVIOR_KEYS = {
    "hold_contract": ["tests", "records", "errors"],
    "replay": ["schema", "frames", "provider", "action_ms", "heads", "actions", "scheduled", "pending_at_clip_end"],
}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def run(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PYTHON), "-B", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def behavior_view(kind: str, data):
    return {key: data.get(key) for key in BEHAVIOR_KEYS[kind]}


def main() -> int:
    parser = argparse.ArgumentParser(description="v0.1 regression suite")
    parser.add_argument("--output", required=True, help="directory for suite artifacts")
    parser.add_argument("--baseline", help="compare behavior artifacts against this baseline directory")
    args = parser.parse_args()
    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    proc = run(["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-q"])
    summary["tests"] = {"returncode": proc.returncode}
    (out / "tests.txt").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")

    proc = run(["tools/hold_contract.py", "--output", str(out / "hold-contract.json")])
    summary["hold_contract"] = {"returncode": proc.returncode}

    proc = run([
        "tools/tap_replay.py",
        "--candidates", str(SYNTHETIC),
        "--annotations", str(SYNTHETIC_ANNOTATIONS),
        "--output", str(out / "replay-synthetic.json"),
    ])
    summary["replay"] = {"returncode": proc.returncode}

    proc = run(["tools/benchmark_taps.py", "--output", str(out / "benchmark-taps.json")])
    summary["benchmark_taps"] = {"returncode": proc.returncode}

    proc = run(["tools/benchmark_tap_recovery.py"])
    recovery = None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                recovery = json.loads(line)
            except json.JSONDecodeError:
                continue
    if recovery is not None:
        (out / "benchmark-recovery.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    summary["benchmark_recovery"] = {"returncode": proc.returncode, "parsed": recovery is not None}

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = [name for name, value in summary.items() if value.get("returncode") != 0]
    if failed:
        print("SUITE FAILED:", ", ".join(failed))
        return 1
    print("suite ok ->", out)

    if args.baseline:
        base = resolve(args.baseline)
        ok = True
        for name, kind in (("hold-contract.json", "hold_contract"), ("replay-synthetic.json", "replay")):
            before = behavior_view(kind, load_json(base / name))
            after = behavior_view(kind, load_json(out / name))
            if before != after:
                ok = False
                print(f"BEHAVIOR DIFF: {name}")
                print("  before:", json.dumps(before, sort_keys=True, ensure_ascii=False)[:400])
                print("  after :", json.dumps(after, sort_keys=True, ensure_ascii=False)[:400])
            else:
                print(f"behavior identical: {name}")
        for bench in ("benchmark-taps.json", "benchmark-recovery.json"):
            before_path, after_path = base / bench, out / bench
            if before_path.is_file() and after_path.is_file():
                before, after = load_json(before_path), load_json(after_path)
                print(f"{bench}: p95 before={before.get('milliseconds', {}).get('p95')} after={after.get('milliseconds', {}).get('p95')}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

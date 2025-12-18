#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sandbox_scripts = "Scripts" if os.name == "nt" else "bin"
VENV_PYTHON = ROOT / ".venv" / sandbox_scripts / ("python.exe" if os.name == "nt" else "python")
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
EXPERIMENTS = [
    ("new_ckks-fl", ROOT / "experiments" / "new_ckks-fl"),
    ("full_ckks-fl", ROOT / "experiments" / "full_ckks-fl"),
    ("selective_ckks-fl", ROOT / "experiments" / "selective_ckks-fl"),
]


def run_pytest(exp_name: str, exp_path: Path) -> int:
    print(f"Running tests for {exp_name}...", flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    prefix = str(exp_path)
    env["PYTHONPATH"] = prefix if not existing else prefix + os.pathsep + existing
    result = subprocess.run([PYTHON, "-m", "pytest"], cwd=exp_path, env=env)
    if result.returncode == 0:
        print(f"{exp_name}: OK\n", flush=True)
    else:
        print(f"{exp_name}: FAILED ({result.returncode})\n", flush=True)
    return result.returncode


def main() -> int:
    statuses = [run_pytest(name, path) for name, path in EXPERIMENTS]
    failed = [name for (name, _), status in zip(EXPERIMENTS, statuses) if status != 0]
    if failed:
        print(f"Tests failed for: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All experiment test suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

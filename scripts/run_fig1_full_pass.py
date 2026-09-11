"""Orchestrates the paper-accurate Fig. 1B/1E full pass (L=10,000s) by invoking
run_fig1_point.py as a FRESH SUBPROCESS per point, so the OS fully reclaims memory
between points -- a single long-lived process was killed by the OOM killer partway
into the Fig. 1E E+I sweep (2026-09-10/11) after 10/15 points had already logged
correct results; Python/numpy don't reliably return freed memory to the OS, so RSS
crept upward across points in that single-process version.

Resumable: skips any (phase, value) point already present in artifacts/metrics.jsonl,
so re-running this script after an interruption picks up where it left off rather than
re-doing the ~29min-per-point work already correctly completed.
"""
import json
import subprocess
import sys

PHASE_VALUES = {
    "fig1b": [0.0, 0.1, 0.2, 0.3, 0.4],
    "fig1e_e_only": [0.0, 0.01, 0.025, 0.05, 0.1],
    "fig1e_e_plus_i": [0.0, 0.01, 0.025, 0.05, 0.1],
}
VALUE_KEY = {"fig1b": "p", "fig1e_e_only": "r_in", "fig1e_e_plus_i": "r_in"}


def already_done(phase: str, value: float) -> bool:
    key = VALUE_KEY[phase]
    log_phase = f"block1_full_pass_{phase}"
    try:
        with open("artifacts/metrics.jsonl") as f:
            for line in f:
                record = json.loads(line)
                if record.get("phase") == log_phase and record.get(key) == value:
                    return True
    except FileNotFoundError:
        pass
    return False


def main():
    for phase, values in PHASE_VALUES.items():
        for value in values:
            if already_done(phase, value):
                print(f"skip {phase} {value} (already logged)")
                continue

            print(f"running {phase} {value} ...")
            result = subprocess.run(
                [sys.executable, "scripts/run_fig1_point.py", phase, str(value)]
            )
            if result.returncode != 0:
                print(f"FAILED {phase} {value} (exit code {result.returncode})")
            else:
                print(f"done {phase} {value}")


if __name__ == "__main__":
    main()

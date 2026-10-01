"""Slurm-array bookkeeping: how a sweep is cut into independent tasks and stitched back
together afterwards.

Every sweep in this project runs as one OS process per task (an in-process pool proved
unreliable on the cluster's ARM64 nodes), so each writes its own row and a later
pass combines them. That shape was duplicated across block 2's two sweeps and block 3's two,
with the same glob-load-sort-write in each; it lives here once instead.

Scheduling is not science: the blocks keep the paper's equations, this keeps the plumbing.
"""
import csv
import glob
import json
import os


def write_task_row(results_dir: str, task_name: str, row: dict) -> None:
    """One task's result, as its own file. Separate files rather than appends because the
    tasks run concurrently and must never contend on one handle.
    """
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/task_{task_name}.json", "w") as f:
        json.dump(row, f)
    print(f"task {task_name} done: {row}", flush=True)


def aggregate_task_rows(results_dir: str, out_csv: str, fields: list[str], sort_key) -> int:
    """Every per-task row combined into one CSV, sorted. Run once the whole array finishes.

    Globs whatever exists rather than requiring a complete set, so a partially-finished
    sweep still aggregates -- which is the normal state here, and why the plots check for
    missing sizes rather than assuming the grid is full.
    """
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/task_*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=sort_key)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def size_repeat_grid(sizes: list[int], repeats: list[int]) -> list[dict]:
    """Flat task list for a sweep over network sizes with a per-size repeat count: one entry
    per realisation, in size order. The flat index is what the Slurm array dispatches on.
    """
    return [{"n": n, "realisation": r} for n, reps in zip(sizes, repeats) for r in range(reps)]

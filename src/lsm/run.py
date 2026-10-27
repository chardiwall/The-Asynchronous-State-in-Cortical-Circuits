"""The LSM extension's one entry point.

    python -m lsm.run calibrate [variant ...]         background input gains (faithful's E rate)
    python -m lsm.run calibrate_task <dataset> [variant ...]   stimulus gains (faithful's task rate)
    python -m lsm.run asynchrony <task_index>         one (variant, N, network) background run
    python -m lsm.run features <dataset> <variant|input> <chunk> [--size N]   one chunk
    python -m lsm.run nchunks <dataset>               chunk count, for the Slurm array
    python -m lsm.run readout <dataset>               train/test every variant's readout
    python -m lsm.run reliability <dataset> <variant> stimulus-locked share of the liquid state
    python -m lsm.run aggregate                       asynchrony task rows -> CSV
    python -m lsm.run plot                            every LSM figure
"""
import json
import math
import os
import sys

import numpy as np

from config import load_config, output_path
from lib.tasks import aggregate_task_rows, write_task_row
from lsm.asynchrony import run_background
from lsm.encode import count_samples, load_samples, subsample_indices

ASYNCHRONY_FIELDS = ["variant", "n", "network_index", "length_s", "rate_excitatory_hz",
                     "rate_inhibitory_hz", "r_bar", "sigma_r", "sigma_r_jittered", "n_pairs_defined",
                     "tracking_EI", "tracking_EX", "c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX",
                     "c_total", "c_EE_T", "c_II_T", "c_XX_T", "c_EI_T", "c_EX_T", "c_IX_T",
                     "c_total_T"]


def asynchrony_grid(config: dict) -> list[dict]:
    """The faithful family at every size (the 1/N test), the other variants and the published
    designs at lsm.size. Designs come last, so the earlier task indices are unchanged."""
    lsm = config["lsm"]
    cells = [("faithful", n) for n in lsm["asynchrony"]["sizes"]]
    cells += [(v, lsm["size"]) for v in (*lsm["variants"], *lsm["designs"]) if v != "faithful"]
    return [{"variant": v, "n": n, "network_index": k}
            for v, n in cells for k in range(lsm["asynchrony"]["n_networks"])]


def task_references(config: dict, name: str) -> tuple[list, int]:
    """(split, index) of every sample used -- fixed random train subset, then test subset."""
    dataset = config["lsm"]["datasets"][name]
    rng = np.random.default_rng(config["seed"])
    train = subsample_indices(count_samples(dataset, "train"), dataset["n_train"], rng)
    test = subsample_indices(count_samples(dataset, "test"), dataset["n_test"], rng)
    return [("train", i) for i in train] + [("test", i) for i in test], len(train)


def _load(dataset: dict, references: list) -> list:
    samples = []
    for split in ("train", "test"):
        samples += load_samples(dataset, split, [i for s, i in references if s == split])
    return samples  # references are train-first, so this keeps their order


def _sized(config: dict, variant: str, argv: list[str]) -> tuple[dict, str]:
    """`--size N` overrides lsm.size; its outputs go to <variant>_N<N> so that sizes coexist."""
    if "--size" not in argv:
        return config, variant
    size = int(argv[argv.index("--size") + 1])
    return {**config, "lsm": {**config["lsm"], "size": size}}, f"{variant}_N{size}"


def features(config: dict, name: str, variant: str, chunk: int, folder: str = None) -> str:
    from lsm.reservoir import chunk_features, thinning

    dataset = config["lsm"]["datasets"][name]
    references, n_train = task_references(config, name)
    size = dataset["chunk"]
    part = references[chunk * size:(chunk + 1) * size]
    # The thinning scalar from a fixed training subset only: no test data, bounded I/O.
    q = thinning(config, dataset, _load(dataset, references[:min(n_train, config["lsm"]["tasks"]["thinning_samples"])]))
    x, y = chunk_features(config, dataset, variant, _load(dataset, part), q, chunk)
    is_train = np.arange(chunk * size, chunk * size + len(part)) < n_train
    out_dir = f"{output_path(config, 'lsm_features_dir')}/{name}/{folder or variant}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/chunk_{chunk:04d}.npz"
    np.savez_compressed(path, x=x, y=y, is_train=is_train, q=q,
                        n_chunks=math.ceil(len(references) / size))
    mean_rate = x.sum() / (x.shape[0] * (x.shape[1] / dataset["n_bins"]) * dataset["window_ms"] / 1000)
    print(f"{path}: {x.shape}, keep q={q:.3f}, mean rate in window {mean_rate:.2f} Hz", flush=True)
    return path


def repeat_reliability(config: dict, name: str, variant: str, folder: str = None) -> dict:
    """The first lsm.reliability.n_samples TEST samples through one variant twice, with
    independent thinning/background noise but the same reservoir: the stimulus-locked
    fraction of the liquid state's variance (lsm.readout.reliability)."""
    from lsm.readout import reliability
    from lsm.reservoir import chunk_features, thinning

    dataset = config["lsm"]["datasets"][name]
    references, n_train = task_references(config, name)
    n = config["lsm"]["reliability"]["n_samples"]
    q = thinning(config, dataset, _load(dataset, references[:min(n_train, config["lsm"]["tasks"]["thinning_samples"])]))
    samples = _load(dataset, references[n_train:n_train + n])
    first, second = (chunk_features(config, dataset, variant, samples, q, seed)[0]
                     for seed in config["lsm"]["reliability"]["noise_streams"])
    row = {"dataset": name, "variant": folder or variant, "n_samples": len(samples),
           "reliability": reliability(first, second)}
    path = f"{output_path(config, 'lsm_features_dir')}/{name}/reliability_{folder or variant}.json"
    with open(path, "w") as f:
        json.dump(row, f)
    print(row, flush=True)
    return row


def readout(config: dict, name: str) -> list[dict]:
    """Every complete variant's readout on one dataset; incomplete variants are reported."""
    import csv
    import glob

    from lsm.readout import train_and_score

    settings = config["lsm"]["readout"]
    rows = []
    for folder in sorted(glob.glob(f"{output_path(config, 'lsm_features_dir')}/{name}/*")):
        chunks = [np.load(p) for p in sorted(glob.glob(f"{folder}/chunk_*.npz"))]
        expected = int(chunks[0]["n_chunks"]) if chunks else 0
        if not chunks or len(chunks) != expected:
            print(f"{folder}: {len(chunks)}/{expected} chunks -- skipped", flush=True)
            continue
        x = np.concatenate([c["x"] for c in chunks]); y = np.concatenate([c["y"] for c in chunks])
        train = np.concatenate([c["is_train"] for c in chunks])
        score = train_and_score(x[train], y[train], x[~train], y[~train], settings["C_grid"],
                                settings["max_iter"], settings["validation_fraction"],
                                np.random.default_rng(config["seed"]))
        rows.append({"dataset": name, "variant": os.path.basename(folder), **score})
        print(rows[-1], flush=True)
    path = output_path(config, "lsm_readout_csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r["dataset"] != name]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "C", "validation_accuracy", "participation_ratio", "train_accuracy",
                                               "test_accuracy", "n_train", "n_test", "n_features"])
        writer.writeheader()
        writer.writerows(old + rows)
    return rows


def main():
    command = sys.argv[1]
    config = load_config("config.yaml")
    if command == "calibrate":
        from lsm.calibrate import calibrate

        gains = calibrate(config, sys.argv[2:])
        path = output_path(config, "lsm_calibration_json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):   # merge, so a subset run keeps the other variants' gains
            with open(path) as f:
                gains = {**json.load(f), **gains}
        with open(path, "w") as f:
            json.dump(gains, f, indent=1)
        print(gains)
    elif command == "calibrate_task":
        from lsm.calibrate import calibrate_task

        dataset = config["lsm"]["datasets"][sys.argv[2]]
        references, _ = task_references(config, sys.argv[2])
        n = config["lsm"]["calibration"]["task_samples"]
        print(calibrate_task(config, dataset, _load(dataset, references[:n]), sys.argv[3:]))
    elif command == "asynchrony":
        task = asynchrony_grid(config)[int(sys.argv[2])]
        row = run_background(config, task["variant"], task["n"], task["network_index"],
                             config["lsm"]["asynchrony"]["length_s"])
        write_task_row(output_path(config, "lsm_asynchrony_dir"),
                       f"{task['variant']}_{task['n']:05d}_{task['network_index']}", row)
    elif command == "nchunks":
        dataset = config["lsm"]["datasets"][sys.argv[2]]
        print(math.ceil(len(task_references(config, sys.argv[2])[0]) / dataset["chunk"]))
    elif command == "features":
        sized, folder = _sized(config, sys.argv[3], sys.argv)
        features(sized, sys.argv[2], sys.argv[3], int(sys.argv[4]), folder)
    elif command == "reliability":
        sized, folder = _sized(config, sys.argv[3], sys.argv)
        repeat_reliability(sized, sys.argv[2], sys.argv[3], folder)
    elif command == "readout":
        readout(config, sys.argv[2])
    elif command == "aggregate":
        n = aggregate_task_rows(output_path(config, "lsm_asynchrony_dir"),
                                output_path(config, "lsm_asynchrony_csv"), ASYNCHRONY_FIELDS,
                                lambda r: (r["variant"], r["n"], r["network_index"]))
        print(f"{n} asynchrony rows")
    elif command == "plot":
        from lsm import plot

        plot.main(config)
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()

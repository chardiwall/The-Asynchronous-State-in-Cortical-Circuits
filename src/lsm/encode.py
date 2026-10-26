"""Event-camera samples -> the X population's input spike train.

Each pooled pixel and polarity is one X neuron (one "channel"), so N_X is the channel count.
A dataset-wide keep-probability thins the events to a target mean channel rate, computed from
the TRAINING split only, and block 3's 2.5 Hz Poisson background stays on every channel so
the reservoir sits in its asynchronous operating point throughout.

Formats read (no tonic needed at run time):
  dvs_gesture  tonic's preprocessed .npy per sample, columns (x, y, polarity, t_ms); the file
               stem is the label (0..10)
  nmnist       one .npz written by lsm.prepare_nmnist: x, y, p, t_us concatenated, plus
               per-sample offsets and labels
"""
import glob
import math
import os

import numpy as np


def to_channels(x, y, p, t_ms, sensor, pool, window_ms):
    """(channel, t_ms) of the events inside [0, window_ms), time-sorted."""
    width_pooled = math.ceil(sensor[0] / pool)
    keep = (t_ms >= 0) & (t_ms < window_ms)
    channel = ((y[keep] // pool) * width_pooled + x[keep] // pool) * 2 + p[keep]
    order = np.argsort(t_ms[keep], kind="stable")
    return channel[order].astype(np.int64), t_ms[keep][order].astype(np.float64)


def n_channels(sensor, pool) -> int:
    return math.ceil(sensor[0] / pool) * math.ceil(sensor[1] / pool) * 2


def keep_probability(mean_events: float, n_channels: int, window_ms: float, target_hz: float) -> float:
    return min(1.0, target_hz * n_channels * window_ms / 1000.0 / mean_events)


def _gesture_files(root: str, split: str) -> list[str]:
    folder = {"train": "ibmGestureTrain", "test": "ibmGestureTest"}[split]
    return sorted(glob.glob(f"{root}/{folder}/*/*.npy"))


def count_samples(dataset: dict, split: str) -> int:
    root = os.path.expanduser(dataset["root"])
    if dataset["format"] == "dvs_gesture":
        return len(_gesture_files(root, split))
    return len(np.load(f"{root}/{split}.npz")["labels"])


def load_samples(dataset: dict, split: str, indices) -> list[tuple]:
    """[(x, y, p, t_ms, label), ...] for the given sample indices only, cropped to the
    window -- the whole split never sits in memory (DVS Gesture alone is ~9GB as float64)."""
    root, window_ms = os.path.expanduser(dataset["root"]), dataset["window_ms"]
    samples = []
    if dataset["format"] == "dvs_gesture":
        files = _gesture_files(root, split)
        for i in indices:
            a = np.load(files[i])
            a = a[a[:, 3] < window_ms]
            samples.append((a[:, 0].astype(np.int64), a[:, 1].astype(np.int64),
                            a[:, 2].astype(np.int64), a[:, 3], int(os.path.basename(files[i])[:-4])))
        return samples
    data = np.load(f"{root}/{split}.npz")
    x, y, p, t_us, bounds, labels = (data[k] for k in ("x", "y", "p", "t_us", "offsets", "labels"))
    for i in indices:
        a, b = bounds[i], bounds[i + 1]
        samples.append((x[a:b].astype(np.int64), y[a:b].astype(np.int64), p[a:b].astype(np.int64),
                        t_us[a:b] / 1000.0, int(labels[i])))
    return samples


def subsample_indices(n_total: int, n, rng: np.random.Generator) -> np.ndarray:
    """A fixed random subset of sample indices, sorted (all of them if n is None/too large)."""
    if n is None or n >= n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, n, replace=False))


def chunk_spike_train(samples, dataset, q, background_hz, burn_in_ms, gap_ms, rng):
    """X spikes for samples laid end to end: burn-in, then (window + gap) per sample.
    Returns (indices, times_ms, onsets_ms, labels). Duplicate (channel, dt-step) pairs are
    dropped, as Brian2's SpikeGeneratorGroup requires at most one spike per step."""
    sensor, pool, window_ms = dataset["sensor"], dataset["pool"], dataset["window_ms"]
    period = window_ms + gap_ms
    onsets = burn_in_ms + period * np.arange(len(samples))
    idx, times = [], []
    for onset, (x, y, p, t, _) in zip(onsets, samples):
        channel, t_ch = to_channels(x, y, p, t, sensor, pool, window_ms)
        keep = rng.random(len(channel)) < q
        idx.append(channel[keep]); times.append(t_ch[keep] + onset)
    total_ms = burn_in_ms + period * len(samples)
    n_ch = n_channels(sensor, pool)
    n_bg = rng.poisson(background_hz * total_ms / 1000.0 * n_ch)
    idx.append(rng.integers(0, n_ch, n_bg)); times.append(rng.uniform(0, total_ms, n_bg))
    idx, times = np.concatenate(idx), np.concatenate(times)
    step = np.round(times / dataset["dt_ms"]).astype(np.int64)
    _, first = np.unique(step * n_ch + idx, return_index=True)
    return idx[first], step[first] * dataset["dt_ms"], onsets, np.array([s[4] for s in samples])

"""Precomputed synaptic current traces, replacing per-event Brian2 delivery.

ds/dt = -s/tau_s + sum_i delta(t-t_i). Over one timestep dt, the homogeneous part decays
exactly by a = exp(-dt/tau_s); spikes landing in that bin add their unit kicks. Binning
pooled spike times into dt-wide bins (counts k[n]) gives the exact recursion:

    s[n] = a*s[n-1] + k[n]

a first-order IIR filter -- scipy.signal.lfilter(b=[1], a=[1,-a], k). This makes the same
dt-grid assumption Brian2's own event timing implies, verified in tests/lib/test_psc.py
against an independent from-scratch reference implementation and (historically, before
model.py's Phase 3.5 rework made it tautological) against Brian2's own SpikeGeneratorGroup
mechanism.
"""
import numpy as np
from scipy.signal import lfilter


def bin_edges(duration_ms: float, dt_ms: float) -> np.ndarray:
    """The dt-wide bin edges synaptic_trace histograms spike times into. Callers making
    several synaptic_trace calls with the same (duration_ms, dt_ms) -- as simulate_pair
    does, once per (cell, population) -- should compute this once and pass it in, rather
    than each call rebuilding an identical array (at production scale, dt_ms=0.05ms over
    L=10,000s, this array alone is ~1.6GB; rebuilding it 4x per simulate_pair call was a
    real, avoidable cost flagged by review).
    """
    n_bins = int(round(duration_ms / dt_ms))
    return np.arange(n_bins + 1) * dt_ms


def synaptic_trace(
    spike_times_ms: list[float],
    duration_ms: float,
    dt_ms: float,
    tau_s_ms: float,
    zi: np.ndarray | None = None,
    bins: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (trace, zf). zf is the filter's final state -- pass it as the next
    chunk's zi to continue the recursion seamlessly across chunk boundaries. Pass a
    precomputed `bins` (from bin_edges) to avoid rebuilding it on every call.
    """
    if bins is None:
        bins = bin_edges(duration_ms, dt_ms)
    counts, _ = np.histogram(spike_times_ms, bins=bins)
    a = np.exp(-dt_ms / tau_s_ms)
    carried_state = 0.0 if zi is None else float(np.asarray(zi).reshape(-1)[0])

    # raw[k] = a*raw[k-1] + counts[k], seeded so raw[-1] == carried_state (the physical
    # trace value at this chunk's start, from the previous chunk). Causality delays a
    # bin's kick to the *next* sample (see module docstring), so the causal trace we
    # return is raw shifted by one and seeded with carried_state -- NOT counts shifted
    # before filtering, which would silently drop the final bin's count entirely (a real
    # bug caught by review: that count never reached the filter, so a spike landing in
    # exactly the last bin of a call vanished from both the trace and the carried-forward
    # state, breaking continuity across chunk boundaries).
    raw, _ = lfilter([1.0], [1.0, -a], counts, zi=[a * carried_state])
    trace = np.concatenate(([carried_state], raw[:-1]))
    zf = np.array([raw[-1]])
    return trace, zf

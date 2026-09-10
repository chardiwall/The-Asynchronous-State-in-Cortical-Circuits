"""ADR 0002: precomputed synaptic current traces, replacing per-event Brian2 delivery.

ds/dt = -s/tau_s + sum_i delta(t-t_i). Over one timestep dt, the homogeneous part decays
exactly by a = exp(-dt/tau_s); spikes landing in that bin add their unit kicks. Binning
pooled spike times into dt-wide bins (counts k[n]) gives the exact recursion:

    s[n] = a*s[n-1] + k[n]

a first-order IIR filter -- scipy.signal.lfilter(b=[1], a=[1,-a], k). This makes the same
dt-grid assumption Brian2's own SpikeGeneratorGroup delivery already does, so it is not a
new source of discretization error (verified in test_current_trace.py against the
existing Brian2 mechanism on a realistic spike train).
"""
import numpy as np
from scipy.signal import lfilter


def synaptic_trace(
    spike_times_ms: list[float],
    duration_ms: float,
    dt_ms: float,
    tau_s_ms: float,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (trace, zf). zf is the filter's final state -- pass it as the next
    chunk's zi to continue the recursion seamlessly across chunk boundaries.
    """
    n_bins = int(round(duration_ms / dt_ms))
    bins = np.arange(n_bins + 1) * dt_ms
    counts, _ = np.histogram(spike_times_ms, bins=bins)
    # Causality: a spike at time t within bin [n*dt, (n+1)*dt) has not happened yet at
    # the sample n*dt itself -- its kick can only appear starting at the *next* grid
    # sample, (n+1)*dt. Shift counts forward by one sample to match (also matches
    # Brian2's own event timing, confirmed by test_matches_existing_brian2_mechanism).
    counts = np.concatenate(([0], counts[:-1]))

    a = np.exp(-dt_ms / tau_s_ms)
    if zi is None:
        zi = np.zeros(1)

    trace, zf = lfilter([1.0], [1.0, -a], counts, zi=zi)
    return trace, zf

"""Seam: synaptic_trace -- the ADR 0002 IIR-filter replacement for per-event Brian2
delivery. s[n] = a*s[n-1] + k[n], a=exp(-dt/tau_s), k[n]=spike count in bin n. Exact
given the same dt-grid assumption Brian2's own SpikeGeneratorGroup already makes.
"""
import numpy as np
import pytest

from block1.current_trace import synaptic_trace

TAU_S_MS = 5.0
DT_MS = 0.01


def test_single_spike_matches_known_exponential_decay():
    # s(t) = exp(-(t-t_spike)/tau_s) for t >= t_spike, 0 before -- the same impulse
    # response used (and independently verified) in test_calibration.py. Causality on a
    # discrete grid: a spike within bin [n*dt,(n+1)*dt) can't affect the sample AT n*dt
    # (hasn't happened yet); its kick first appears at the next grid sample, (n+1)*dt.
    trace, _ = synaptic_trace(
        spike_times_ms=[10.0], duration_ms=50.0, dt_ms=DT_MS, tau_s_ms=TAU_S_MS,
    )
    bin_of_spike = int(10.0 / DT_MS)  # 1000
    effective_kick_index = bin_of_spike + 1  # 1001, i.e. t=10.01ms
    n = np.arange(len(trace))
    expected = np.where(
        n >= effective_kick_index,
        np.exp(-(n - effective_kick_index) * DT_MS / TAU_S_MS),
        0.0,
    )
    assert trace == pytest.approx(expected, abs=1e-6)


def test_zi_zf_chaining_matches_a_single_unchunked_call():
    # The chunking mechanism Phase 3.5 needs: filtering in two pieces with zf->zi carried
    # forward must equal filtering the whole thing at once.
    spikes = [5.0, 12.0, 12.0, 30.0, 45.0, 45.0, 60.0]
    duration_ms = 80.0

    whole, _ = synaptic_trace(spikes, duration_ms, DT_MS, TAU_S_MS)

    chunk_1, zf = synaptic_trace(
        [s for s in spikes if s < 40.0], 40.0, DT_MS, TAU_S_MS
    )
    chunk_2, _ = synaptic_trace(
        [s - 40.0 for s in spikes if s >= 40.0], duration_ms - 40.0, DT_MS, TAU_S_MS, zi=zf,
    )
    chained = np.concatenate([chunk_1, chunk_2])

    assert chained == pytest.approx(whole, abs=1e-9)


def test_matches_independent_from_scratch_reference():
    # Independent reference: a plain per-sample Python loop (no lfilter, no Brian2), so
    # a bug in the lfilter-based implementation can't also be present here by
    # construction. Replaces a Brian2-comparison test that became tautological once
    # model.py's Phase 3.5 rework made simulate_pair call synaptic_trace internally --
    # at that point both sides of that comparison were the same function, so it could
    # no longer catch a real regression (caught by review).
    rng = np.random.default_rng(99)
    duration_ms = 200.0
    spikes = np.sort(rng.uniform(0, duration_ms - 1, 40))

    trace, _ = synaptic_trace(spikes.tolist(), duration_ms, DT_MS, TAU_S_MS)

    n_bins = int(round(duration_ms / DT_MS))
    a = np.exp(-DT_MS / TAU_S_MS)
    counts = np.zeros(n_bins, dtype=int)
    for s in spikes:
        counts[int(s / DT_MS)] += 1
    reference = np.zeros(n_bins)
    for n in range(1, n_bins):
        reference[n] = a * reference[n - 1] + counts[n - 1]

    assert trace == pytest.approx(reference, abs=1e-9)


def test_spike_in_final_bin_is_carried_forward_not_dropped():
    # Regression for a real bug caught by review: a spike landing in exactly the last
    # dt-bin of a call must still flow into zf (the carried-forward state) even though
    # it can't show up in THIS call's own trace (its effect starts at the next sample,
    # which is the next chunk's sample 0). The old implementation shifted counts via
    # `counts[:-1]` before filtering, which discarded the final bin's count entirely --
    # it reached neither the trace nor zf, silently losing the spike across a chunk
    # boundary despite the docstring's explicit "continue seamlessly" promise.
    duration_ms = 40.0
    spike_in_last_bin = 39.995  # bin index 3999 of 4000 -- the final bin

    chunk_1, zf = synaptic_trace([spike_in_last_bin], duration_ms, DT_MS, TAU_S_MS)
    chunk_2, _ = synaptic_trace([], duration_ms, DT_MS, TAU_S_MS, zi=zf)

    whole, _ = synaptic_trace([spike_in_last_bin], 2 * duration_ms, DT_MS, TAU_S_MS)

    assert np.concatenate([chunk_1, chunk_2]) == pytest.approx(whole, abs=1e-9)
    assert chunk_2[0] > 0.0  # the spike's effect must appear at the next chunk's start

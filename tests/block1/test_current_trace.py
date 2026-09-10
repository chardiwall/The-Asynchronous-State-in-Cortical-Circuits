"""Seam: synaptic_trace -- the ADR 0002 IIR-filter replacement for per-event Brian2
delivery. s[n] = a*s[n-1] + k[n], a=exp(-dt/tau_s), k[n]=spike count in bin n. Exact
given the same dt-grid assumption Brian2's own SpikeGeneratorGroup already makes.
"""
import numpy as np
import pytest

from block1.current_trace import synaptic_trace
from block1.model import simulate_pair

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


def test_matches_existing_brian2_mechanism_on_realistic_spike_train():
    # Validation required by PROGRESS.md 3.4: the new precomputed-filter mechanism must
    # agree with simulate_pair's existing (soon to be replaced) SpikeGeneratorGroup
    # mechanism on the same input, before Phase 3.5 trusts it to take over. E-only,
    # j_e_mV=1.0 makes i_syn_a_mV == s_E directly (i_syn = J_E*s_E - J_I*s_I, s_I=0).
    rng = np.random.default_rng(99)
    duration_ms = 200.0
    spikes = np.sort(rng.uniform(0, duration_ms - 1, 40)).tolist()

    trace, _ = synaptic_trace(spikes, duration_ms, DT_MS, TAU_S_MS)

    brian2_result = simulate_pair(
        e_spikes_a=spikes, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=1.0, j_i_mV=1.0,
        tau_m_ms=10.0, tau_s_ms=TAU_S_MS, theta_mV=20.0,
        v_reset_mV=10.0, t_ref_ms=2.0, duration_ms=duration_ms, dt_ms=DT_MS,
    )

    assert trace == pytest.approx(brian2_result.i_syn_a_mV, abs=1e-2)

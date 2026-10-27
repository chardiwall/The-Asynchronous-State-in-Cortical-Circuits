"""Seam: lsm.designs -- the three published liquids (Maass 2002, Kaiser 2017, Wijesinghe
2019). Truth: Woo et al. 2024's Table 1 link counts, Kaiser's released fixed in-degrees,
Markram et al. 1998's closed-form depression/facilitation recursion, and each neuron
equation's own rest point and PSC decay."""
import brian2 as b2
import numpy as np
import pytest

from config import load_config
from lsm.asynchrony import run_background
from lsm.designs import build_design, stp_synapses

CONFIG = load_config("config.yaml")
SILENT = {"kind": "spike_generator", "indices": np.array([], dtype=int), "times_ms": np.array([])}


def _build(name, external=SILENT, n_external=578, seed=0):
    return build_design(CONFIG, name, n_external, external, np.random.default_rng(seed))


def test_wijesinghe_links_match_the_table_and_have_no_inhibitory_recurrence():
    net = _build("wijesinghe2019")
    # Table 1: E->E 63 840, E->I 16 000, I->E 20 000, I->I 0; 5 binomial SDs of tolerance
    for pair, count, pairs_possible, p in (("E_to_E", 63840, 400 * 399, 0.4),
                                           ("E_to_I", 16000, 400 * 100, 0.4),
                                           ("I_to_E", 20000, 100 * 400, 0.5)):
        assert abs(len(net.synapses[pair]) - count) < 5 * np.sqrt(pairs_possible * p * (1 - p))
    assert "I_to_I" not in net.synapses and "X_to_I" not in net.synapses
    ee = net.synapses["E_to_E"]
    assert not np.any(np.asarray(ee.i) == np.asarray(ee.j))


def test_kaiser_every_neuron_receives_two_excitatory_and_one_inhibitory_synapse():
    net = _build("kaiser2017")
    # Table 1: E->E 800, E->I 200, I->E 400, I->I 100 -- NEST fixed_indegree 2 (E) and 1 (I)
    for pair, n_post, k in (("E_to_E", 400, 2), ("E_to_I", 100, 2), ("I_to_E", 400, 1), ("I_to_I", 100, 1)):
        assert np.all(np.bincount(np.asarray(net.synapses[pair].j), minlength=n_post) == k)
    # each input channel projects to exactly 100 excitatory neurons, none inhibitory
    assert np.all(np.bincount(np.asarray(net.synapses["X_to_E"].i), minlength=578) == 100)
    assert "X_to_I" not in net.synapses


def test_maass_column_has_the_reviews_link_total_and_maass_type_ratios():
    net = _build("maass2002")
    counts = {p: len(net.synapses[p]) for p in ("E_to_E", "E_to_I", "I_to_E", "I_to_I")}
    assert sum(counts.values()) == pytest.approx(3630, rel=0.08)   # Table 1: 3630 +- 15
    # Maass's C: I->E (0.4) is twice E->I (0.2) over the same 400 x 100 pairs
    assert counts["I_to_E"] / counts["E_to_I"] == pytest.approx(2.0, rel=0.25)


def test_stp_jumps_follow_markrams_recursion():
    # Three presynaptic spikes 20 ms apart into a non-decaying accumulator: each jump is
    # w * u_n * R_n with u_n = U + u_{n-1}(1-U)e^{-gap/F}, R_n = 1 + (R_{n-1}(1-u_{n-1}) - 1)e^{-gap/D}
    U, D, F, gap = 0.5, 1100.0, 50.0, 20.0
    b2.start_scope()
    pre = b2.SpikeGeneratorGroup(1, [0, 0, 0], [10, 30, 50] * b2.ms)
    post = b2.NeuronGroup(1, "dacc/dt = 0*nA/ms : amp", method="euler")
    syn = stp_synapses(pre, post, "acc", ([0], [0]), w=np.array([1.0]) * b2.nA,
                       stp={"U": U, "D": D, "F": F}, delay_ms=np.array([0.0]), name="stp_test")
    mon = b2.StateMonitor(post, "acc", record=0, dt=1 * b2.ms)
    b2.Network(pre, post, syn, mon).run(60 * b2.ms)
    acc = np.asarray(mon.acc[0] / b2.nA)
    jumps = np.diff(acc[[5, 20, 40, 59]])
    u, r, expected = 0.0, 1.0, []
    for n in range(3):
        u_n = U + u * (1 - U) * np.exp(-gap / F) if n else U
        r_n = 1 + (r * (1 - u) - 1) * np.exp(-gap / D) if n else 1.0
        expected.append(u_n * r_n)
        u, r = u_n, r_n
    np.testing.assert_allclose(jumps, expected, rtol=1e-3)


def test_maass_neurons_rest_below_threshold_without_input():
    net = _build("maass2002")
    net.network.run(300 * b2.ms)
    assert net.spikes_e.num_spikes == 0 and net.spikes_i.num_spikes == 0
    # R_m I_back = 1 MOhm * 13.5 nA = 13.5 mV, 10 tau_m after a start in [13.5, 15]
    np.testing.assert_allclose(np.asarray(net.group_e.V / b2.mV), 13.5, atol=0.01)


@pytest.mark.parametrize("name, variable, tau_ms", [("maass2002", "I_X", 3.0),
                                                     ("wijesinghe2019", "g_X", 1.0)])
def test_one_input_spike_gives_an_exponential_psc_of_the_synaptic_weight(name, variable, tau_ms):
    external = {"kind": "spike_generator", "indices": np.array([0]), "times_ms": np.array([5.0])}
    net = _build(name, external)
    syn = net.synapses["X_to_E"]
    targets, weights = np.asarray(syn.j), np.asarray(syn.w[:])
    cell = int(targets[np.flatnonzero(np.asarray(syn.i) == 0)[0]])
    mon = b2.StateMonitor(net.group_e, variable, record=cell, dt=0.05 * b2.ms)
    net.network.add(mon)
    net.network.run((5.0 + 0.15 + tau_ms) * b2.ms)
    trace = np.asarray(getattr(mon, variable)[0])
    w = weights[(np.asarray(syn.i) == 0) & (targets == cell)].sum()   # weights already carry the gain
    assert w > 0 and trace.max() == pytest.approx(w, rel=1e-6)
    peak = int(np.argmax(trace))
    assert trace[peak + int(round(tau_ms / 0.05))] / trace[peak] == pytest.approx(np.exp(-1), rel=0.01)


def test_input_weights_carry_the_calibrated_gain():
    gain_net = build_design(CONFIG, "maass2002", 578, SILENT, np.random.default_rng(0), gain_key="task_input_gain")
    doubled = {**CONFIG, "lsm": {**CONFIG["lsm"], "task_input_gain": {"maass2002": 2.0}}}
    net = build_design(doubled, "maass2002", 578, SILENT, np.random.default_rng(0), gain_key="task_input_gain")
    ratio = np.asarray(net.synapses["X_to_E"].w[:]) / np.asarray(gain_net.synapses["X_to_E"].w[:])
    np.testing.assert_allclose(ratio, 2.0 / CONFIG["lsm"]["task_input_gain"]["maass2002"])


def test_a_design_runs_as_a_reservoir_and_gives_one_feature_row_per_sample():
    from lsm.reservoir import chunk_features

    rng = np.random.default_rng(0)
    def sample(label):   # 3000 random events over the 300 ms window of a 34x34 sensor
        n = 3000
        return (rng.integers(0, 34, n), rng.integers(0, 34, n), rng.integers(0, 2, n),
                np.sort(rng.uniform(0, 300, n)), label)
    dataset = CONFIG["lsm"]["datasets"]["nmnist"]
    x, y = chunk_features(CONFIG, dataset, "kaiser2017", [sample(3), sample(7)], q=1.0, chunk_index=0)
    assert x.shape == (2, CONFIG["lsm"]["size"] * dataset["n_bins"]) and list(y) == [3, 7]
    assert x.sum() > 0 and np.all(np.isfinite(x))


def test_design_background_run_gives_an_exact_current_decomposition():
    # gain pinned at 1 (strongly driven), so the check does not depend on the calibration
    driven = {**CONFIG, "lsm": {**CONFIG["lsm"], "input_gain": {**CONFIG["lsm"]["input_gain"], "maass2002": 1.0}}}
    row = run_background(driven, "maass2002", n=500, network_index=0, length_s=1.0)
    assert row["variant"] == "maass2002" and row["rate_excitatory_hz"] > 0
    parts = row["c_EE"] + row["c_II"] + row["c_XX"] + 2 * (row["c_EI"] + row["c_EX"] + row["c_IX"])
    assert np.isclose(parts, row["c_total"])

"""Seams for Fig. 3C-D (SOM S-p.21): disable_spiking -- the "spiking mechanism had been
disabled" step, done by raising only the recorded cells' threshold out of reach;
recorded_pairs -- the pair enumeration whose counts must reproduce the paper's stated 450
and 1000 (docs/adr/0006); signed_peak -- Fig. 3D's "peak height of the membrane potential
CCG", which must stay NEGATIVE for the anticorrelated EPSP-IPSP condition.
"""
import brian2 as b2
import numpy as np
import pytest

from block3.measure import recorded_pairs, signed_peak
from block3.figures_cd import disable_spiking


def _group(n=4):
    eqs = "dV/dt = -V / (10*ms) : volt\ntheta : volt (constant)"
    group = b2.NeuronGroup(n, eqs, threshold="V>theta", reset="V=0*mV", method="exact")
    group.theta = -50 * b2.mV
    return group


def test_disabling_raises_only_the_named_cells_threshold():
    group = _group()

    disable_spiking(group, [1, 2])

    theta_mV = np.asarray(group.theta / b2.mV)
    assert theta_mV[0] == pytest.approx(-50.0)
    assert theta_mV[3] == pytest.approx(-50.0)
    assert theta_mV[1] > 1e3   # unreachable for a membrane potential in mV
    assert theta_mV[2] > 1e3


def test_a_disabled_cell_actually_stops_spiking_in_a_real_run():
    """The assertion above only checks the STORED threshold. This one checks the property
    that Fig. 3C-D depends on: after disable_spiking, a driven cell emits no spikes at all,
    so V integrates freely with no reset. If Brian2 ever inlined a (constant) parameter into
    generated code, the per-index assignment would silently no-op and this test would catch
    it -- the stored-value test would not.
    """
    b2.start_scope()
    group = b2.NeuronGroup(
        4, "dV/dt = (100*mV - V) / (5*ms) : volt\ntheta : volt (constant)",
        threshold="V>theta", reset="V=0*mV", method="exact")
    group.theta = 20 * b2.mV
    group.V = 0 * b2.mV
    disable_spiking(group, [1, 2])
    monitor = b2.SpikeMonitor(group)

    b2.Network(group, monitor).run(50 * b2.ms)

    fired = set(np.asarray(monitor.i).tolist())
    assert fired == {0, 3}, f"disabled cells must not spike, but {fired} fired"


def test_pair_counts_reproduce_the_papers_450_and_1000():
    # docs/adr/0006: 10 cells per condition per network, 10 networks.
    within_a, within_b, cross = recorded_pairs(n_a=10, n_b=10)

    assert len(within_a) == 45 and len(within_b) == 45   # 45 * 10 networks = 450
    assert len(cross) == 100                              # 100 * 10 networks = 1000


def test_pairs_are_distinct_and_never_self_paired():
    within_a, _, cross = recorded_pairs(n_a=4, n_b=3)

    assert all(i != j for i, j in within_a)
    assert len(set(within_a)) == len(within_a)
    assert len(cross) == 12


def test_signed_peak_of_a_positive_correlogram_is_its_maximum():
    assert signed_peak(np.array([0.1, 0.6, 0.2])) == pytest.approx(0.6)


def test_signed_peak_stays_negative_for_an_anticorrelated_pair():
    # The EPSP-IPSP (gold) condition is large and NEGATIVE; taking a plain max would
    # report 0.1 here and silently flip Fig. 3D's V-shape into a flat line.
    assert signed_peak(np.array([0.1, -0.8, 0.05])) == pytest.approx(-0.8)


def test_signed_peak_ignores_undefined_lags():
    assert signed_peak(np.array([np.nan, 0.4, np.nan])) == pytest.approx(0.4)


def test_signed_peak_is_undefined_when_every_lag_is():
    assert np.isnan(signed_peak(np.array([np.nan, np.nan])))


def test_grid_covers_every_level_plus_the_cross_condition(tmp_path):
    from block3.figures_cd import build_task_grid

    config = {"spiking_network": {"fig3cd": {
        "i_app_nA": [-1.3, 0.0, 3.7], "n_networks": 2,
        "epsp_condition_nA": -1.3, "ipsp_condition_nA": 3.7}}}

    grid = build_task_grid(config)

    # Per network: one same-condition task per level, plus ONE cross task for Fig. 3C's
    # gold EPSP-IPSP curve. 3 levels + 1 cross = 4 tasks, times 2 networks.
    assert len(grid) == 8
    assert {(t["i_app_a_nA"], t["i_app_b_nA"]) for t in grid} == {
        (-1.3, -1.3), (0.0, 0.0), (3.7, 3.7), (-1.3, 3.7)}
    assert sorted({t["network_index"] for t in grid}) == [0, 1]


def test_signed_peak_takes_the_central_lobe_not_a_larger_side_lobe():
    # Fig. 3D's published axis is entirely non-negative with a minimum near +0.02, so the
    # intermediate holding potentials have SMALL central peaks. A global extremum over the
    # whole +/-50 ms window would return the -0.4 side-lobe here and flip the point
    # negative, breaking the V-shape. The central lobe is the peak.
    ccg = np.array([-0.4, -0.1, 0.05, 0.02, -0.3])

    assert signed_peak(ccg) == pytest.approx(0.05)


def test_signed_peak_keeps_a_negative_central_lobe():
    # The EPSP-IPSP condition is genuinely negative at zero lag; the lobe logic must not
    # "correct" that into the positive side-lobe.
    ccg = np.array([0.1, -0.30, -0.42, -0.25, 0.09])

    assert signed_peak(ccg) == pytest.approx(-0.42)

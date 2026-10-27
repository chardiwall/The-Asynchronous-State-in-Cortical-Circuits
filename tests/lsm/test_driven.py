"""Seam: lsm.driven.repeat_correlations -- correlations of a liquid while it is DRIVEN,
split by two repeats of the same stimuli (independent input noise, same reservoir) into a
stimulus-locked (signal) part and a trial-to-trial (noise) part. Truth: synthetic counts
built as a shared signal plus independent noise, each with a known pairwise correlation."""
import numpy as np
import pytest

from lsm.driven import repeat_correlations


def _equicorrelated(rng, rho, cells, bins):
    c = np.full((cells, cells), rho) + (1 - rho) * np.eye(cells)
    return np.linalg.cholesky(c) @ rng.normal(size=(cells, bins))


def test_repeat_split_recovers_known_signal_and_noise_correlations():
    rng = np.random.default_rng(0)
    cells, bins, rho_s, rho_n = 40, 20000, 0.3, -0.02
    signal = _equicorrelated(rng, rho_s, cells, bins)
    first = signal + _equicorrelated(rng, rho_n, cells, bins)
    second = signal + _equicorrelated(rng, rho_n, cells, bins)
    r = repeat_correlations(first, second)
    assert r["r_signal"] == pytest.approx(rho_s, abs=0.01)
    assert r["r_noise"] == pytest.approx(rho_n, abs=0.005)
    # equal signal and noise variance: the total is their average
    assert r["r_total"] == pytest.approx((rho_s + rho_n) / 2, abs=0.01)


def test_silent_cells_are_left_out_rather_than_poisoning_the_mean():
    rng = np.random.default_rng(1)
    first = rng.poisson(2.0, size=(5, 1000)).astype(float)
    second = rng.poisson(2.0, size=(5, 1000)).astype(float)
    first[0] = second[0] = 0.0
    r = repeat_correlations(first, second)
    # independent repeats carry no stimulus-locked variance, so r_signal may be undefined
    assert np.isfinite(r["r_total"]) and np.isfinite(r["r_noise"])


def test_a_driven_run_reports_exact_total_and_noise_current_decompositions():
    from config import load_config
    from lsm.driven import driven_row

    config = load_config("config.yaml")
    rng = np.random.default_rng(0)
    def sample(label):   # 3000 random events over the 300 ms window of a 34x34 sensor
        return (rng.integers(0, 34, 3000), rng.integers(0, 34, 3000), rng.integers(0, 2, 3000),
                np.sort(rng.uniform(0, 300, 3000)), label)
    row = driven_row(config, config["lsm"]["datasets"]["nmnist"], "maass2002", [sample(1), sample(2), sample(3)], q=1.0)
    for suffix in ("_T", "_noise"):
        c = {k: row[f"c_{k}{suffix}"] for k in ("EE", "II", "XX", "EI", "EX", "IX", "total")}
        assert np.isclose(c["EE"] + c["II"] + c["XX"] + 2 * (c["EI"] + c["EX"] + c["IX"]), c["total"])
    assert row["rate_excitatory_hz"] > 0 and np.isfinite(row["r_total"]) and np.isfinite(row["r_noise"])

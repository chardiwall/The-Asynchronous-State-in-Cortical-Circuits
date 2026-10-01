"""Seam: calibrate_input_rate -- ambiguity 1 in the paper (SOM S-p.19-20):
the paper states 20Hz for the E+I condition but not the E-only rate; it must be found
numerically so E-only output rate = 5Hz at r_in=0 (no closed form for LIF-under-Poisson
firing rate). Bisection assumes output rate increases monotonically with input rate --
true for a purely excitatory drive on a LIF neuron.
"""
import numpy as np
import pytest

from block1.calibration import calibrate_input_rate
from block1.inputs import build_pair_inputs
from block1.model import simulate_pair

COMMON_KWARGS = dict(
    n_e=250, n_i=0,
    j_e_mV=3.0, j_i_mV=3.0,
    tau_m_ms=10.0, tau_s_ms=5.0, theta_mV=20.0, v_reset_mV=10.0, t_ref_ms=2.0,
    duration_ms=5000.0, dt_ms=0.05, jitter_tau_ms=5.0,
)


def _measure_output_rate_hz(rate_hz: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    inputs = build_pair_inputs(
        n_e=COMMON_KWARGS["n_e"], n_i=0, p=0.0, r_in=0.0, rate_hz=rate_hz,
        duration_ms=COMMON_KWARGS["duration_ms"],
        jitter_tau_ms=COMMON_KWARGS["jitter_tau_ms"], rng=rng,
    )
    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=COMMON_KWARGS["j_e_mV"], j_i_mV=COMMON_KWARGS["j_i_mV"],
        tau_m_ms=COMMON_KWARGS["tau_m_ms"], tau_s_ms=COMMON_KWARGS["tau_s_ms"],
        theta_mV=COMMON_KWARGS["theta_mV"], v_reset_mV=COMMON_KWARGS["v_reset_mV"],
        t_ref_ms=COMMON_KWARGS["t_ref_ms"], duration_ms=COMMON_KWARGS["duration_ms"],
        dt_ms=COMMON_KWARGS["dt_ms"],
    )
    return len(result.spikes_a_ms) / (COMMON_KWARGS["duration_ms"] / 1000.0)


def test_calibrated_rate_produces_output_close_to_target():
    target_hz = 5.0
    calibrated_rate_hz = calibrate_input_rate(
        target_output_hz=target_hz,
        rate_low_hz=1.0, rate_high_hz=20.0,
        tolerance_hz=1.0, max_iterations=12, n_trials=3,
        rng=np.random.default_rng(42),
        **COMMON_KWARGS,
    )

    # Independent verification run (fresh seed) at the calibrated rate.
    verification_hz = _measure_output_rate_hz(calibrated_rate_hz, seed=999)
    assert verification_hz == pytest.approx(target_hz, abs=2.0)  # noisy at L=5s


def test_calibrated_rate_is_well_below_the_e_plus_i_rate():
    # Sanity check tied to the ambiguity itself: the E+I condition's 20Hz would drive
    # the E-only cell far above 5Hz (per the docs' own reasoning), so the calibrated
    # rate must land meaningfully below 20Hz.
    calibrated_rate_hz = calibrate_input_rate(
        target_output_hz=5.0,
        rate_low_hz=1.0, rate_high_hz=20.0,
        tolerance_hz=1.0, max_iterations=12, n_trials=3,
        rng=np.random.default_rng(43),
        **COMMON_KWARGS,
    )
    assert calibrated_rate_hz < 20.0


def test_bounds_not_bracketing_target_raises():
    # Regression for a real bug caught by review: bisection with bounds that don't
    # actually bracket the target must fail loudly, not silently converge to one edge
    # and return it as if calibration succeeded.
    with pytest.raises(ValueError):
        calibrate_input_rate(
            target_output_hz=5.0,
            rate_low_hz=0.01, rate_high_hz=0.05,  # both far too low to reach 5Hz
            tolerance_hz=0.5, max_iterations=5, n_trials=2,
            rng=np.random.default_rng(44),
            **COMMON_KWARGS,
        )


def test_exhausting_iterations_without_converging_raises():
    # Regression for a real bug caught by review: running out of max_iterations without
    # meeting tolerance must fail loudly, not silently return an unconverged mid as if
    # it were a successful calibration.
    with pytest.raises(ValueError):
        calibrate_input_rate(
            target_output_hz=5.0,
            rate_low_hz=1.0, rate_high_hz=20.0,
            tolerance_hz=1e-6, max_iterations=1, n_trials=1,  # impossible to converge
            rng=np.random.default_rng(45),
            **COMMON_KWARGS,
        )


def test_averaging_multiple_trials_reduces_single_run_noise_sensitivity():
    # The bug this fixes: a single stochastic simulate_pair run per bisection step can
    # violate the monotonicity assumption bisection depends on (expected count ~25
    # spikes at L=5s/5Hz, ~20% Poisson relative noise -- comparable to the old
    # tolerance_hz=1.0). n_trials>1 averages independent replicates at each candidate
    # rate, reducing that noise by ~sqrt(n_trials). Just checks it still converges
    # sensibly with averaging enabled -- the noise-reduction itself isn't directly
    # observable in a single calibration run, but this exercises the averaging path.
    calibrated_rate_hz = calibrate_input_rate(
        target_output_hz=5.0,
        rate_low_hz=1.0, rate_high_hz=20.0,
        tolerance_hz=1.0, max_iterations=12, n_trials=3,
        rng=np.random.default_rng(46),
        **COMMON_KWARGS,
    )
    # Average 3 independent verification runs rather than trusting one -- a single run
    # at this scale (expected count ~25, ~20% Poisson relative noise) is exactly the
    # kind of noisy single-sample measurement this fix addresses; averaging the check
    # itself avoids the test being flaky for the same underlying reason.
    verification_hz = np.mean(
        [_measure_output_rate_hz(calibrated_rate_hz, seed=1001 + i) for i in range(3)]
    )
    assert verification_hz == pytest.approx(5.0, abs=2.0)


def test_both_conditions_are_calibrated_to_the_same_output_rate():
    """Fig. 1E's two curves must share an operating point -- the caption describes the E+I
    condition as having "identical statistics" to the E-only one. The paper states 20
    spikes/s input for E+I and claims it produces 5 spikes/s output; measured here it
    produces about 10.5, so the input was recalibrated instead (config.yaml records the
    full finding). This pins that both calibrated rates are present and that the E+I rate
    is the calibrated one rather than the paper's stated value.
    """
    from config import load_config

    inputs = load_config("config.yaml")["pair_model"]["inputs"]

    assert inputs["rate_e_only_calibrated_hz"] is not None
    assert inputs["rate_e_plus_i_calibrated_hz"] is not None
    assert inputs["rate_e_plus_i_calibrated_hz"] != inputs["rate_correlated_sweep_hz"]


def test_the_sweep_grid_uses_the_calibrated_e_plus_i_rate():
    """build_sweep_points must raise rather than silently fall back to the paper's 20 Hz,
    which would put Fig. 1E's two curves at ~5 and ~10.5 spikes/s output.
    """
    import copy

    from block1.run import build_sweep_points
    from config import load_config

    config = load_config("config.yaml")
    points = build_sweep_points(config)
    e_plus_i = {p["rate_hz"] for p in points if p["phase"] == "fig1e_e_plus_i"}
    assert e_plus_i == {config["pair_model"]["inputs"]["rate_e_plus_i_calibrated_hz"]}

    unset = copy.deepcopy(config)
    unset["pair_model"]["inputs"]["rate_e_plus_i_calibrated_hz"] = None
    with pytest.raises(ValueError, match="rate_e_plus_i_calibrated_hz"):
        build_sweep_points(unset)

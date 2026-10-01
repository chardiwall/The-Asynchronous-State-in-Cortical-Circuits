"""Block 1's two calibrations: the values the paper specifies indirectly, or not at all.

**Synaptic weights.** A single presynaptic spike gives s(t) = exp(-t/tau_s), so the
subthreshold response to one input is the standard double-exponential PSP. J is solved in
closed form so that V(t_peak) equals the stated PSP peak (S-p.19: +0.75 mV EPSP, 0.75 mV
IPSP magnitude -- both J_E and J_I are positive, the membrane equation's own sign structure
supplies the inhibitory sign). simulate_psp_peak checks that closed form against the real
pair model rather than against a separate hand-built reference.

**Input rates**, by bisection, since a LIF neuron's output rate under Poisson bombardment
has no closed form. Two conditions need it, for different reasons:

- *E-only*: the paper never states this rate at all (ambiguity 1).
- *E+I*: the paper states 20 spikes/s and says it "produce[s] an output rate of 5 spikes/s
  when r_in = 0" (S-p.19-20). Measured, 20 spikes/s gives ~10.5. The weight calibration, the
  analytic mean drive and the free-membrane statistics each check out independently, so the
  paper's two numbers are mutually inconsistent here. Decision (2026-09-30): honour the
  stated OUTPUT rate and recalibrate the input, as the E-only ambiguity was already
  resolved, so both Fig. 1E curves share an operating point.

Monotonicity holds for a purely excitatory drive and still holds for E+I here, since
N_E > N_I keeps the mean drive positive while the variance also grows with rate. Each
measurement is a noisy Monte Carlo estimate, so n_trials averages replicates: the tolerance
must sit above the resulting standard error or bisection decides on noise (config.yaml's
calibration_rate stanza records the sizing).

Run via the block's entry point: python -m block1.run calibrate {e_only,e_plus_i}
"""
import math
import numpy as np

from block1.inputs import build_pair_inputs
from block1.model import simulate_pair

def psp_weight(tau_m_ms: float, tau_s_ms: float, target_peak_mV: float) -> float:
    if tau_s_ms >= tau_m_ms:
        raise ValueError(
            f"psp_weight requires tau_m_ms > tau_s_ms (double-exponential PSP is only "
            f"valid then); got tau_m_ms={tau_m_ms}, tau_s_ms={tau_s_ms}"
        )
    t_peak_ms = (tau_m_ms * tau_s_ms) / (tau_m_ms - tau_s_ms) * math.log(tau_m_ms / tau_s_ms)
    peak_per_unit_j = (tau_s_ms / (tau_m_ms - tau_s_ms)) * (
        math.exp(-t_peak_ms / tau_m_ms) - math.exp(-t_peak_ms / tau_s_ms)
    )
    return target_peak_mV / peak_per_unit_j


def simulate_psp_peak(
    j_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    dt_ms: float,
    duration_ms: float,
) -> float:
    """Drive cell A of the actual Phase 2 pair model (simulate_pair) with a single E
    spike at t=0, weight j_mV, and return the observed peak V in mV. Reuses simulate_pair
    directly (rather than a separate hand-built one-neuron model) so this check can never
    silently drift from the dynamics Phase 2 actually uses -- j_i_mV=j_mV is passed too
    but has no effect here since i_spikes_a is empty.
    """
    result = simulate_pair(
        e_spikes_a=[0.0],
        i_spikes_a=[],
        e_spikes_b=[],
        i_spikes_b=[],
        j_e_mV=j_mV,
        j_i_mV=j_mV,
        tau_m_ms=tau_m_ms,
        tau_s_ms=tau_s_ms,
        theta_mV=theta_mV,
        v_reset_mV=v_reset_mV,
        t_ref_ms=t_ref_ms,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
    )
    return float(result.v_a_mV.max())


def calibrate_synaptic_weights(config: dict) -> tuple[float, float]:
    """J_E, J_I (both positive mV magnitudes) from config.yaml's pair_model section."""
    tau_m_ms = config["pair_model"]["neuron"]["tau_m_ms"]
    tau_s_ms = config["pair_model"]["synapse"]["tau_s_ms"]
    epsp_peak_mV = config["pair_model"]["synapse"]["epsp_peak_mV"]
    ipsp_peak_mV = config["pair_model"]["synapse"]["ipsp_peak_mV"]

    if epsp_peak_mV <= 0:
        raise ValueError(f"epsp_peak_mV must be positive, got {epsp_peak_mV}")
    if ipsp_peak_mV >= 0:
        raise ValueError(f"ipsp_peak_mV must be negative, got {ipsp_peak_mV}")

    j_e = psp_weight(tau_m_ms, tau_s_ms, epsp_peak_mV)
    j_i = psp_weight(tau_m_ms, tau_s_ms, -ipsp_peak_mV)
    return j_e, j_i


def calibrate_input_rate(
    target_output_hz: float,
    n_e: int,
    n_i: int,
    j_e_mV: float,
    j_i_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    duration_ms: float,
    dt_ms: float,
    jitter_tau_ms: float,
    rate_low_hz: float,
    rate_high_hz: float,
    tolerance_hz: float,
    max_iterations: int,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    def single_trial_rate(trial_rate_hz: float) -> float:
        inputs = build_pair_inputs(
            n_e=n_e, n_i=n_i, p=0.0, r_in=0.0, rate_hz=trial_rate_hz,
            duration_ms=duration_ms, jitter_tau_ms=jitter_tau_ms, rng=rng,
        )
        result = simulate_pair(
            e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
            e_spikes_b=[], i_spikes_b=[],
            j_e_mV=j_e_mV, j_i_mV=j_i_mV,
            tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms, theta_mV=theta_mV,
            v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms,
            duration_ms=duration_ms, dt_ms=dt_ms,
        )
        return len(result.spikes_a_ms) / (duration_ms / 1000.0)

    def output_rate_at(trial_rate_hz: float) -> float:
        return float(np.mean([single_trial_rate(trial_rate_hz) for _ in range(n_trials)]))

    low, high = rate_low_hz, rate_high_hz
    rate_at_low = output_rate_at(low)
    rate_at_high = output_rate_at(high)
    if not (rate_at_low <= target_output_hz <= rate_at_high):
        raise ValueError(
            f"[rate_low_hz={low}, rate_high_hz={high}] does not bracket "
            f"target_output_hz={target_output_hz}: measured rates are "
            f"[{rate_at_low}, {rate_at_high}]. Widen the search bounds."
        )

    mid, rate = (low + high) / 2.0, None
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        rate = output_rate_at(mid)
        if abs(rate - target_output_hz) < tolerance_hz:
            return mid
        if rate < target_output_hz:
            low = mid
        else:
            high = mid

    raise ValueError(
        f"calibrate_input_rate did not converge within {max_iterations} "
        f"iterations: last candidate {mid}Hz measured {rate}Hz against target "
        f"{target_output_hz}Hz (tolerance {tolerance_hz}Hz). Increase max_iterations, "
        f"n_trials, or duration_ms."
    )


def calibrate_from_config(condition: str, config: dict) -> float:
    """calibrate_input_rate driven straight from config.yaml, for either condition.

    The sixteen arguments below are all config lookups; collecting them here rather than at
    the call site keeps the caller to one line and puts the knowledge of which settings this
    calibration needs in the module that owns the calibration.
    """
    pair = config["pair_model"]
    neuron, inputs_cfg, search = pair["neuron"], pair["inputs"], pair["calibration_rate"]
    j_e, j_i = calibrate_synaptic_weights(config)
    return calibrate_input_rate(
        target_output_hz=5.0,       # SOM S-p.19-20 states this output rate for r_in = 0
        n_e=inputs_cfg["n_excitatory"],
        n_i=inputs_cfg["n_inhibitory"] if condition == "e_plus_i" else 0,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=pair["synapse"]["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=search["duration_ms"],
        dt_ms=pair["simulation"]["dt_ms"],
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"],
        rate_low_hz=search["rate_low_hz"], rate_high_hz=search["rate_high_hz"],
        tolerance_hz=search["tolerance_hz"], max_iterations=search["max_iterations"],
        n_trials=search["n_trials"], rng=np.random.default_rng(config["seed"]),
    )

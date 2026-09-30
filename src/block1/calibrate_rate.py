"""Input-rate calibration by bisection: finds the input rate that drives the cell to a
target output rate, since there is no closed form for a LIF neuron's output rate under
Poisson bombardment. Assumes output rate increases monotonically with input rate -- true
for a purely excitatory drive, and still true for the E+I condition here because
N_E > N_I leaves the mean drive positive while the variance also grows with rate.

Two conditions need it, for different reasons:

- **E-only** (ambiguity 1, docs/paper/01-postsynaptic-pair.md): the paper never states the
  E-only input rate at all.
- **E+I**: the paper DOES state 20 spikes/s, and states that it "produce[s] an output rate
  of 5 spikes/s when r_in = 0" (SOM S-p.19-20) -- but measured with this repo's parameters
  20 spikes/s produces 12.05 +- 2.57 spikes/s, not 5. The PSP calibration, the analytic
  mean drive and the free-membrane statistics each check out independently, so the paper's
  two numbers are not consistent with each other here. The researcher chose (2026-09-30) to
  honour the stated OUTPUT rate and recalibrate the input, matching how the E-only
  ambiguity was already resolved, so that both Fig. 1E curves share an operating point --
  which is what the paper means by "identical statistics".

Run it: python -m block1.calibrate_rate {e_only,e_plus_i}

Each output-rate measurement is itself a noisy Monte Carlo estimate (a single
simulate_pair run's spike count is Poisson-ish), which can violate bisection's
monotonicity assumption if the noise is comparable to tolerance_hz -- e.g. at L=5s,
target=5Hz, expected count ~25 has ~20% relative noise. n_trials averages independent
replicates at each candidate rate to reduce this by ~sqrt(n_trials) (a real bug caught
by review: the original single-sample version could converge to a systematically wrong
rate with no signal that anything had gone wrong).
"""
import numpy as np

from block1.dataset import build_pair_inputs
from block1.model import simulate_pair


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


def main():
    """Derives one condition's input rate and prints it. The result goes into config.yaml
    by hand, with the run's settings recorded next to it -- nothing here writes config.
    """
    import sys

    from block1.calibration import calibrate_synaptic_weights
    from config import load_config

    condition = sys.argv[1]
    config = load_config("config.yaml")
    pair = config["pair_model"]
    neuron, inputs_cfg = pair["neuron"], pair["inputs"]
    j_e, j_i = calibrate_synaptic_weights(config)

    rate = calibrate_input_rate(
        target_output_hz=5.0,          # SOM S-p.19-20 states this output rate for r_in = 0
        n_e=inputs_cfg["n_excitatory"],
        n_i=inputs_cfg["n_inhibitory"] if condition == "e_plus_i" else 0,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=pair["synapse"]["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"],
        duration_ms=pair["calibration_rate"]["duration_ms"],
        dt_ms=pair["simulation"]["dt_ms"],
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"],
        rate_low_hz=pair["calibration_rate"]["rate_low_hz"],
        rate_high_hz=pair["calibration_rate"]["rate_high_hz"],
        tolerance_hz=pair["calibration_rate"]["tolerance_hz"],
        max_iterations=pair["calibration_rate"]["max_iterations"],
        n_trials=pair["calibration_rate"]["n_trials"],
        rng=np.random.default_rng(config["seed"]),
    )
    print(f"{condition}: input rate {rate:.5f} Hz gives ~5 Hz output at r_in = 0.\n"
          f"Put it in config.yaml and record the settings used.")


if __name__ == "__main__":
    main()

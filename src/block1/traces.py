"""Fig. 1C/1F: illustrative example traces at p=0.2, r_in=0.025 (the black/blue circles
of Fig. 1E). Purely visual/qualitative -- no new measurement, reuses simulate_pair's
already-tested output plus current_trace.synaptic_trace directly for Fig. 1F's separate
E/I current components (avoids touching simulate_pair's tested interface).
"""
import matplotlib.pyplot as plt
import numpy as np

from block1.calibration import calibrate_synaptic_weights
from block1.current_trace import synaptic_trace
from block1.dataset import build_pair_inputs
from block1.model import simulate_pair


def plot_fig1c_e_only(config: dict, duration_ms: float, rng: np.random.Generator, out_path: str):
    """E-only: raster + combined current + V_m for both cells."""
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    p = config["pair_model"]["sweeps"]["p_fixed"]
    r_in = config["pair_model"]["sweeps"]["r_in_fig1f_example"]

    j_e, j_i = calibrate_synaptic_weights(config)

    inputs = build_pair_inputs(
        n_e=inputs_cfg["n_excitatory"], n_i=0, p=p, r_in=r_in,
        rate_hz=inputs_cfg["rate_e_only_calibrated_hz"], duration_ms=duration_ms,
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"], rng=rng,
    )
    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=[], e_spikes_b=inputs.e_spikes_b,
        i_spikes_b=[], j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms,
        dt_ms=config["pair_model"]["simulation"]["dt_ms"],
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].eventplot([result.spikes_a_ms, result.spikes_b_ms], lineoffsets=[1, 0], linelengths=0.8)
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["cell B", "cell A"])
    axes[0].set_title(f"Fig. 1C (E-only, p={p}, r_in={r_in}): raster")
    axes[1].plot(result.t_ms, result.i_syn_a_mV, label="cell A")
    axes[1].plot(result.t_ms, result.i_syn_b_mV, label="cell B", alpha=0.7)
    axes[1].set_ylabel("I_syn (mV)"); axes[1].legend(loc="upper right")
    axes[2].plot(result.t_ms, result.v_a_mV, label="cell A")
    axes[2].plot(result.t_ms, result.v_b_mV, label="cell B", alpha=0.7)
    axes[2].set_ylabel("V (mV)"); axes[2].set_xlabel("t (ms)"); axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_fig1f_e_plus_i(config: dict, duration_ms: float, rng: np.random.Generator, out_path: str):
    """E+I: V_m for both cells, plus cell A's E and I current components shown
    SEPARATELY (not combined) to show simultaneous excursions -- the paper's point.
    """
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    p = config["pair_model"]["sweeps"]["p_fixed"]
    r_in = config["pair_model"]["sweeps"]["r_in_fig1f_example"]
    dt_ms = config["pair_model"]["simulation"]["dt_ms"]

    j_e, j_i = calibrate_synaptic_weights(config)

    inputs = build_pair_inputs(
        n_e=inputs_cfg["n_excitatory"], n_i=inputs_cfg["n_inhibitory"], p=p, r_in=r_in,
        rate_hz=inputs_cfg["rate_correlated_sweep_hz"], duration_ms=duration_ms,
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"], rng=rng,
    )
    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
        e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms, dt_ms=dt_ms,
    )
    # Separate E/I components for cell A only (reuses the same pooled arrivals; doesn't
    # touch simulate_pair's tested interface).
    s_e_a, _ = synaptic_trace(inputs.e_spikes_a, duration_ms, dt_ms, synapse["tau_s_ms"])
    s_i_a, _ = synaptic_trace(inputs.i_spikes_a, duration_ms, dt_ms, synapse["tau_s_ms"])
    i_e_a = j_e * s_e_a
    i_i_a = -j_i * s_i_a

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].eventplot([result.spikes_a_ms, result.spikes_b_ms], lineoffsets=[1, 0], linelengths=0.8)
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["cell B", "cell A"])
    axes[0].set_title(f"Fig. 1F (E+I, p={p}, r_in={r_in}): raster")
    axes[1].plot(result.t_ms, i_e_a, label="E current (cell A)", color="tab:red")
    axes[1].plot(result.t_ms, i_i_a, label="I current (cell A)", color="tab:blue")
    axes[1].set_ylabel("I (mV)"); axes[1].legend(loc="upper right")
    axes[2].plot(result.t_ms, result.v_a_mV, label="cell A")
    axes[2].plot(result.t_ms, result.v_b_mV, label="cell B", alpha=0.7)
    axes[2].set_ylabel("V (mV)"); axes[2].set_xlabel("t (ms)"); axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

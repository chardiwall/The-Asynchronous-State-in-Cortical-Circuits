"""Fig. 3C-D: the intracellular cancellation experiment (SOM S-p.21), the paper's
experimentally testable prediction.

Protocol: constant current is injected into cell pairs whose spiking mechanism has been
disabled, the current levels chosen to isolate EPSPs and IPSPs near their respective
reversal potentials or to mix them at intermediate potentials, and cross-correlograms of
the two voltages are averaged over 450 pairs from ten networks at 50 s each -- except the
EPSP-IPSP condition (Fig. 3C's gold curve), which used 1000 pairs.

One task = one network at one ORDERED pair of current levels. Ten E cells are held at each
level with spiking disabled, and every within-A, within-B and cross pair's CCG is averaged.
With a == b that gives Fig. 3D's point for that level (45 pairs x 10 networks = 450); with
a = the EPSP level and b = the IPSP level it gives Fig. 3C's gold curve (100 x 10 = 1000).
Ten cells per level is DERIVED, not stated: it is the only simple scheme reproducing both
of the SOM's pair counts (see config.yaml: fig3cd.n_recorded_cells_per_condition). Cells
with spiking disabled stop driving their targets, and that perturbation grows with the
number of conditions recorded at once, so one task runs ONE condition pair rather than all
eight levels at once.

Usage (one Slurm array task): python -m block3.figures_cd <task_index>
Usage (one explicit run):     python -m block3.figures_cd <network_index> <i_app_a> <i_app_b>
Usage (aggregate):            python -m block3.figures_cd --aggregate
"""
import os

import brian2 as b2
import numpy as np

from analysis import lagged_correlation
from block3.measure import recorded_pairs, sample_neuron_subset, signed_peak
from block3.model import build_network

UNREACHABLE_THRESHOLD_mV = 1e6  # far above any voltage this conductance-based model reaches


def disable_spiking(group, indices) -> None:
    """The SOM's "spiking mechanism had been disabled": these cells' firing threshold is
    raised out of reach, so V never crosses it, no reset fires, and the trace is the free
    integration of the summed synaptic input -- which is what the recording is for.

    model.py declares theta as a per-neuron parameter precisely so this touches only the
    recorded cells; every other neuron keeps the config threshold and spikes normally.
    """
    group.theta[list(indices)] = UNREACHABLE_THRESHOLD_mV * b2.mV





def build_task_grid(config: dict) -> list[dict]:
    """One task per (network, condition). Per network: every swept level run against
    itself, which gives Fig. 3D's point for that level and Fig. 3C's green/red/black
    curves, plus ONE cross task at (EPSP level, IPSP level) for Fig. 3C's gold curve.

    The cross task also yields both of its own levels' same-condition pairs, so the two
    same-condition tasks at those levels are redundant and could be dropped; they are kept
    so that every level is produced by one uniform kind of task.
    """
    fig3cd = config["spiking_network"]["fig3cd"]
    conditions = [(level, level) for level in fig3cd["i_app_nA"]]
    conditions.append((fig3cd["epsp_condition_nA"], fig3cd["ipsp_condition_nA"]))
    return [{"network_index": n, "i_app_a_nA": a, "i_app_b_nA": b}
            for n in range(fig3cd["n_networks"]) for a, b in conditions]


def run_one_task(network_index: int, i_app_a_nA: float, i_app_b_nA: float, config: dict) -> dict:
    net_config = config["spiking_network"]
    fig3cd = net_config["fig3cd"]
    n_each = fig3cd["n_recorded_cells_per_condition"]
    burn_in_ms = net_config["burn_in_ms"]
    duration_ms = fig3cd["length_s"] * 1000.0

    rng = np.random.default_rng(config["seed"] + network_index)
    net = build_network({**net_config, "dt_ms": net_config["simulation"]["dt_ms"]}, rng)

    recorded = sample_neuron_subset(net_config["populations"]["n_excitatory"], 2 * n_each, rng)
    disable_spiking(net.group_e, recorded)
    net.group_e.I_app[list(recorded[:n_each])] = i_app_a_nA * b2.nA
    net.group_e.I_app[list(recorded[n_each:])] = i_app_b_nA * b2.nA

    # This block's own CCG bin, not analysis.bin_dt_ms and not analysis.ccg.bin_dt_ms:
    # those two mean the spike-train binning and the in vivo spike-CCG bin respectively, so
    # borrowing either left one quantity with two settings and silently ignored the sibling.
    # The paper states no bin for the voltages. Sampling V more coarsely than the 0.05 ms
    # integration step does not affect the integration.
    bin_ms = fig3cd["ccg_bin_ms"]
    # `record` order is PRESERVED, not sorted (verified against Brian2 2.10.1's source:
    # StateMonitor stores np.asarray(record) with no sort). recorded_pairs' "A-cells are
    # rows 0..n_each-1" depends on that, and sample_neuron_subset returns unsorted output.
    # Do not "tidy up" by sorting `recorded` -- it would scramble the A/B split silently.
    monitor = b2.StateMonitor(net.group_e, "V", record=list(recorded), dt=bin_ms * b2.ms)
    net.network.add(monitor)
    net.network.run(duration_ms * b2.ms)

    t_ms = np.asarray(monitor.t / b2.ms)
    v_mV = np.asarray(monitor.V / b2.mV)[:, t_ms >= burn_in_ms]
    assert not np.any(np.isnan(v_mV)), f"NaN in network {network_index}'s recorded V"

    max_lag = int(round(fig3cd["ccg_max_lag_ms"] / bin_ms))
    holding_mV = v_mV.mean(axis=1)
    within_a, within_b, cross = recorded_pairs(n_each, n_each)

    row = {"network_index": network_index, "i_app_a_nA": i_app_a_nA, "i_app_b_nA": i_app_b_nA,
           "bin_ms": bin_ms, "max_lag_ms": fig3cd["ccg_max_lag_ms"],
           "holding_a_mV": float(holding_mV[:n_each].mean()),
           "holding_b_mV": float(holding_mV[n_each:].mean())}
    for name, pairs in (("within_a", within_a), ("within_b", within_b), ("cross", cross)):
        ccg = np.nanmean([lagged_correlation(v_mV[i], v_mV[j], max_lag) for i, j in pairs], axis=0)
        row[f"n_pairs_{name}"] = len(pairs)
        row[f"ccg_{name}"] = [float(v) for v in ccg]
        row[f"peak_{name}"] = signed_peak(ccg)
    return row


def aggregate_vm_ccg(results_dir: str, out_csv: str) -> int:
    """Per-task JSON -> one long-form CSV, a line per (task, pair group, lag). That is what
    block3.plot reads for both 3C's curves and 3D's peak-versus-holding-potential points.

    Not lib.tasks.aggregate_task_rows: that writes one line per task, and these rows each
    carry a whole correlogram that has to be unrolled.
    """
    import csv
    import glob
    import json

    rows = []
    for path in sorted(glob.glob(f"{results_dir}/task_*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: (r["i_app_a_nA"], r["i_app_b_nA"], r["network_index"]))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["network_index", "i_app_a_nA", "i_app_b_nA", "group",
                         "holding_mV", "n_pairs", "peak", "lag_ms", "ccg"])
        for r in rows:
            for group, holding in (("within_a", r["holding_a_mV"]),
                                    ("within_b", r["holding_b_mV"]),
                                    ("cross", 0.5 * (r["holding_a_mV"] + r["holding_b_mV"]))):
                ccg = r[f"ccg_{group}"]
                lags = (np.arange(len(ccg)) - len(ccg) // 2) * r["bin_ms"]
                for lag, value in zip(lags, ccg):
                    writer.writerow([r["network_index"], r["i_app_a_nA"], r["i_app_b_nA"],
                                     group, holding, r[f"n_pairs_{group}"],
                                     r[f"peak_{group}"], lag, value])
    print(f"{out_csv}: {len(rows)} tasks")
    return len(rows)

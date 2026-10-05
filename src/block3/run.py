"""Block 3's one entry point. Every Fig. 3 run is a subcommand here.

    python -m block3.run networks <index> [--seconds N]   Fig. 3A-B statistics, one network
    python -m block3.run panels [--seconds N]             Fig. 3A-B panel data, one network
    python -m block3.run vm <task_index>                  Fig. 3C-D, one (network, condition)
    python -m block3.run vm <network> <i_app_a> <i_app_b> the same, stated explicitly
    python -m block3.run aggregate                        per-task rows -> CSVs
    python -m block3.run plot                             all four panels

A short `--seconds` run is the smoke test: the same code path at the same paper-scale N over
a shorter window, so it is a true subset of the real run rather than a second implementation.
"""
import os
import sys

import brian2 as b2
import numpy as np

from block3 import figures_ab, figures_cd
from block3.model import build_network
from config import load_config, output_path
from lib.tasks import aggregate_task_rows, write_task_row

NETWORK_FIELDS = ["network_index", "rate_excitatory_hz", "rate_inhibitory_hz",
                  "r_bar_EE", "nan_free"]


def parse_length_s(argv: list[str], default: float) -> float:
    """The `--seconds N` override, parsed strictly.

    A malformed invocation RAISES rather than falling back to `default`. The flag exists so a
    short smoke test can precede a Slurm array whose tasks cost many hours each, and a silent
    fallback would turn that check into the very run it was meant to de-risk, while appearing
    to succeed.
    """
    if len(argv) <= 2:
        return default
    if argv[2] != "--seconds":
        raise ValueError(f"unrecognised argument {argv[2]!r}; the only option is --seconds N")
    if len(argv) < 4:
        raise ValueError("--seconds needs a value, e.g. --seconds 5")
    return float(argv[3])


def main():
    command = sys.argv[1]
    config = load_config("config.yaml")
    net_config = config["spiking_network"]

    if command == "networks":
        index = int(sys.argv[2])
        duration_ms = parse_length_s(
            ["run", "-", *sys.argv[3:]], net_config["simulation"]["length_s"]) * 1000.0
        write_task_row(output_path(config, "block3_networks_dir"), f"{index:04d}",
                       figures_ab.run_one_task(index, config, duration_ms))

    elif command == "panels":
        # Fig. 3B's own length, not the statistics run's: the surrogate histogram's width is
        # pure estimator noise, so too short a run makes the null as wide as the measurement.
        length_s = parse_length_s(["run", "-", *sys.argv[2:]], net_config["panels"]["length_s"])
        duration_ms = length_s * 1000.0

        # Checked BEFORE the network is built: the failure would otherwise land after a
        # multi-day simulation with nothing written.
        projected_gb = figures_ab.check_rate_matrix_fits(config, length_s)
        os.makedirs(output_path(config, "block3_panels_dir"), exist_ok=True)
        print(f"Fig. 3B rate matrix will be ~{projected_gb:.2f}GB", flush=True)

        rng = np.random.default_rng(config["seed"])
        net = build_network({**net_config, "dt_ms": net_config["simulation"]["dt_ms"]}, rng)
        net.network.run(duration_ms * b2.ms)

        spikes = figures_ab.population_spikes(net, config, net_config["burn_in_ms"])
        recorded_ms = duration_ms - net_config["burn_in_ms"]
        figures_ab.write_fig3a(spikes, recorded_ms, config, rng)
        figures_ab.write_fig3b(spikes, recorded_ms, config, rng)
        print(f"Fig. 3A/3B data written to "
              f"{output_path(config, 'block3_panels_dir')}/ ({length_s}s run)")

    elif command == "vm":
        if len(sys.argv) > 4:
            index, a, b = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        else:
            task = figures_cd.build_task_grid(config)[int(sys.argv[2])]
            index, a, b = task["network_index"], task["i_app_a_nA"], task["i_app_b_nA"]
        row = figures_cd.run_one_task(index, a, b, config)
        write_task_row(output_path(config, "block3_vm_ccg_dir"), f"{index:02d}_{a}_{b}", row)

    elif command == "aggregate":
        networks = aggregate_task_rows(output_path(config, "block3_networks_dir"),
                                       output_path(config, "block3_networks_csv"),
                                       NETWORK_FIELDS, lambda r: r["network_index"])
        print(f"{networks} network rows")
        figures_cd.aggregate_vm_ccg(output_path(config, "block3_vm_ccg_dir"),
                                    output_path(config, "block3_vm_ccg_csv"))

    elif command == "plot":
        from block3 import plot

        plot.main(config)

    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()

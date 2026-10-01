"""Fig. 2C's current-correlation components (main text p.588, M-Eq 3): the decomposition
of c into c_EE, c_II, c_XX, c_EI, c_EX, c_IX over the E/I/X COMPONENTS of each cell's
afferent current. Walks the same (N, realisation) grid as block2.full_pass and reuses its
dispatch/aggregation seam; the only difference is that each task records a subsampled
continuous current-component trace (fast_model.simulate_fast_current) instead of the full
population's binary state.

Averaged over pairs of E cells (the same population Fig. 2G's histogram uses). The paper
says only "across cell pairs" and does not name the population.

Usage (one task): python -m block2.full_pass_current <task_index>
Usage (after the array finishes): python -m block2.full_pass_current --aggregate
"""
import sys

from block2.eval import current_component_correlations
from lib.glauber import simulate_fast_current
from block2.full_pass import aggregate, task_parameters, write_task

RESULTS_DIR = "artifacts/block2_full_pass_current"
FIELDS = ["n", "realisation", "c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX", "c_total"]


def run_one_current_task(
    n: int, realisation: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int, subsample_size: int,
) -> dict:
    components = simulate_fast_current(n=n, p=p, j=j, m_x=m_x, theta=theta,
                                        length_tau=length_tau, sampling_rate=sampling_rate,
                                        burn_in_tau=burn_in_tau, seed=seed + realisation,
                                        subsample_size=subsample_size)
    return {"n": n, "realisation": realisation, **current_component_correlations(components)}


def main():
    out_csv = "artifacts/block2_full_pass_current.csv"
    if sys.argv[1] == "--aggregate":
        print(f"wrote {aggregate(RESULTS_DIR, out_csv, FIELDS)} rows to {out_csv}")
        return

    task_index = int(sys.argv[1])
    config, net, j, task = task_parameters(task_index)
    row = run_one_current_task(
        n=task["n"], realisation=task["realisation"], p=net["connection_probability"], j=j,
        m_x=net["m_x"], theta=net["theta"], length_tau=net["full_pass"]["length_tau"],
        sampling_rate=net["sampling_rate_instantaneous"], burn_in_tau=net["burn_in_tau"],
        seed=config["seed"], subsample_size=net["fig2g_subsample_neurons"],
    )
    write_task(RESULTS_DIR, task_index, row)


if __name__ == "__main__":
    main()

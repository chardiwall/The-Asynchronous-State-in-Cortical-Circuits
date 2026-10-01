"""Seams: build_full_pass_grid -- maps a flat Slurm array task index to its
(N, realisation) job, one entry per repeat per size (config.yaml's
binary_network.full_pass); run_one_task -- runs ONE realisation via the tested
simulate_fast_one and returns its summary row (rates + the five S-Eq 28-29
correlations), the per-task unit of work each Slurm array task does.
"""
from block2.run import run_one_task
from lib.tasks import size_repeat_grid

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}


def test_grid_has_one_entry_per_repeat_per_size_in_order():
    grid = size_repeat_grid(sizes=[10, 20], repeats=[2, 3])

    assert grid == [
        {"n": 10, "realisation": 0},
        {"n": 10, "realisation": 1},
        {"n": 20, "realisation": 0},
        {"n": 20, "realisation": 1},
        {"n": 20, "realisation": 2},
    ]


def test_run_one_task_returns_the_expected_summary_fields():
    row = run_one_task(n=5, realisation=2, p=0.5, j=J, m_x=0.3, theta=0.5,
                        length_tau=3, sampling_rate=1, burn_in_tau=1, seed=0)

    assert row["n"] == 5
    assert row["realisation"] == 2
    for key in ("rate_E", "rate_I", "rate_X", "r_EE", "r_II", "r_EI", "r_EX", "r_IX"):
        assert key in row

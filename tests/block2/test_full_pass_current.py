"""Seam: run_one_current_task -- runs ONE realisation via simulate_fast_current
and returns c_EE, c_II, c_EI (S-Eq 32-33) plus their sum c = c_EE+c_II+2c_EI
(M-Eq 3), the per-task unit of work for block2.full_pass_current's Slurm array.
"""
from block2.full_pass_current import run_one_current_task

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}


def test_run_one_current_task_returns_the_expected_fields_and_consistent_total():
    row = run_one_current_task(n=20, realisation=1, p=0.5, j=J, m_x=0.3, theta=0.5,
                                length_tau=5, sampling_rate=1, burn_in_tau=1, seed=0,
                                subsample_size=8)

    assert row["n"] == 20
    assert row["realisation"] == 1
    for key in ("c_EE", "c_II", "c_EI", "c_total"):
        assert key in row
    assert row["c_total"] == row["c_EE"] + row["c_II"] + 2 * row["c_EI"]

"""Seams: the Python-level wrappers around panel_traces' JIT recordings --
build connectivity + initial state, matching simulate_fast_current's pattern.
"""
from block2.panel_traces import (
    population_mean_trace,
    single_cell_components_trace,
    subsample_state_trace,
)

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}


def test_population_mean_trace_shape():
    result = population_mean_trace(n=20, p=0.5, j=J, m_x=0.3, theta=0.5,
                                    window_tau=2, sampling_rate=4, burn_in_tau=1, seed=0)
    assert result.shape == (3, 8)  # window_tau * sampling_rate samples


def test_subsample_state_trace_shape():
    result = subsample_state_trace(n=20, p=0.5, j=J, m_x=0.3, theta=0.5,
                                    window_tau=2, sampling_rate=1, burn_in_tau=1,
                                    seed=0, subsample_size=5)
    assert result.shape == (5, 2)


def test_single_cell_components_trace_shape():
    result = single_cell_components_trace(n=20, p=0.5, j=J, m_x=0.3, theta=0.5,
                                           window_tau=2, sampling_rate=4, burn_in_tau=1,
                                           seed=0, cell_index=3)
    assert result.shape == (4, 8)

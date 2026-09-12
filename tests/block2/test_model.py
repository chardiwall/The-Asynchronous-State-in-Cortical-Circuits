"""Seams: afferent_current -- S-Eq(7), the net input that decides whether a picked
recurrent neuron fires; tick -- the exact simulation algorithm's single elementary
step (S-p.18): pick one of 3N neurons uniformly, apply its population's update rule.
Weights below are hand-built (not build_weights) so the expected h_i is computable
independently of the code under test.
"""
import numpy as np

from block2.model import afferent_current, tick

THETA = 1.0

STATE = {
    "E": np.array([1, 0, 1]),
    "I": np.array([0, 1, 0]),
    "X": np.array([1, 1, 0]),
}

WEIGHTS = {
    "EE": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float),
    "EI": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=float),
    "EX": np.array([[1, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=float),
    "IE": np.zeros((3, 3)),
    "II": np.zeros((3, 3)),
    "IX": np.zeros((3, 3)),
}


def test_current_exactly_at_threshold_is_zero_not_positive():
    # h_0 = EE[0].E + EI[0].I + EX[0].X - theta = 0 + 0 + 1 - 1 = 0
    h = afferent_current(STATE, WEIGHTS, THETA, population="E", i=0)
    assert h == 0.0


def test_current_above_threshold_is_positive():
    # h_1 = EE[1].E + EI[1].I + EX[1].X - theta = 1 + (-1) + 2 - 1 = 1
    h = afferent_current(STATE, WEIGHTS, THETA, population="E", i=1)
    assert h == 1.0


def make_state() -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in STATE.items()}


class FixedRng:
    """Stub in place of np.random.Generator: returns pre-set answers, not a real
    stream, so tick's picked-index/redraw arithmetic is tested independently of
    numpy's RNG implementation."""
    def __init__(self, pick: int, draw: float = 0.0):
        self._pick = pick
        self._draw = draw

    def integers(self, low: int, high: int) -> int:
        assert low <= self._pick < high
        return self._pick

    def random(self) -> float:
        return self._draw


def test_tick_sets_a_recurrent_neuron_by_thresholding_its_current():
    state = make_state()
    # index 1 -> E-population local index 1 (population block order E,I,X; n=3)
    # h_1 = 1 (see test_current_above_threshold_is_positive) -> fires
    population, i, new_value = tick(state, WEIGHTS, THETA, m_x=0.5, rng=FixedRng(pick=1))

    assert (population, i, new_value) == ("E", 1, 1)
    assert state["E"][1] == 1


def test_tick_redraws_an_external_neuron_from_its_bernoulli_rate():
    state = make_state()
    # index 8 -> X-population local index 2 (E block [0,3), I block [3,6), X block [6,9))
    population, i, new_value = tick(state, WEIGHTS, THETA, m_x=0.5, rng=FixedRng(pick=8, draw=0.3))

    assert (population, i, new_value) == ("X", 2, 1)  # draw 0.3 < m_x 0.5 -> active
    assert state["X"][2] == 1


def test_tick_external_neuron_inactive_when_draw_exceeds_rate():
    state = make_state()
    population, i, new_value = tick(state, WEIGHTS, THETA, m_x=0.5, rng=FixedRng(pick=6, draw=0.7))

    assert (population, i, new_value) == ("X", 0, 0)  # draw 0.7 > m_x 0.5 -> inactive
    assert state["X"][0] == 0

"""Seam: _afferent_current_jit -- the Numba nopython-mode version of S-Eq(7)'s
afferent current (model.afferent_current), restructured onto one concatenated
(3n,) state vector and two stacked (n,3n) weight matrices (weights_E = [EE|EI|EX],
weights_I = [IE|II|IX]) since nopython mode can't take a dict of named arrays.
Same hand-built values as test_model.py's afferent_current tests, laid out
concatenated, so the expected h_i is the same independently-computed number.
"""
import numpy as np

from block2.fast_model import _afferent_current_jit

THETA = 1.0

# E=[1,0,1], I=[0,1,0], X=[1,1,0], concatenated
STATE = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=np.float64)

# weights_E row i = [EE[i] | EI[i] | EX[i]], same values as test_model.py's WEIGHTS
WEIGHTS_E = np.array([
    [0, 1, 0, -1, 0, 0, 1, 0, 0],  # EE[0]=[0,1,0], EI[0]=[-1,0,0], EX[0]=[1,0,0]
    [0, 0, 1, 0, -1, 0, 2, 0, 0],  # EE[1]=[0,0,1], EI[1]=[0,-1,0], EX[1]=[2,0,0]
], dtype=np.float64)


def test_current_exactly_at_threshold_is_zero_not_positive():
    # h_0 = EE[0].E + EI[0].I + EX[0].X - theta = 0 + 0 + 1 - 1 = 0
    h = _afferent_current_jit(STATE, WEIGHTS_E[0], THETA)
    assert h == 0.0


def test_current_above_threshold_is_positive():
    # h_1 = EE[1].E + EI[1].I + EX[1].X - theta = 1 + (-1) + 2 - 1 = 1
    h = _afferent_current_jit(STATE, WEIGHTS_E[1], THETA)
    assert h == 1.0

"""Seam: exact_current_correlation / approx_meq1_current_correlation -- the analytic
predictions Phase 8 cross-checks simulated c against (main text ref. 15, M-Eq 1)."""
import pytest

from analysis import approx_meq1_current_correlation, exact_current_correlation


def test_exact_reduces_to_p_when_r_in_zero():
    # No correlated component -> pure shared-fraction correlation, c=p exactly.
    assert exact_current_correlation(p=0.2, r_in=0.0, n=250) == pytest.approx(0.2)


def test_exact_matches_hand_computed_value():
    # p=0.2, r_in=0.025, N=250: numerator=0.2+0.025*249.8=6.445,
    # denominator=1+0.025*249=7.225, c=6.445/7.225=0.89204...
    c = exact_current_correlation(p=0.2, r_in=0.025, n=250)
    assert c == pytest.approx(6.445 / 7.225, rel=1e-6)


def test_approx_matches_exact_when_p_and_r_in_n_are_small():
    # M-Eq(1) validity condition: p ~ r_in*N << 1.
    exact = exact_current_correlation(p=0.01, r_in=0.0002, n=10)
    approx = approx_meq1_current_correlation(p=0.01, r_in=0.0002, n=10)
    assert approx == pytest.approx(exact, rel=0.05)


def test_approx_diverges_from_exact_when_r_in_n_not_small():
    # At r_in=0.025, N=250 (this project's Fig. 1E point): r_in*N=6.25, NOT << 1 --
    # M-Eq(1) is invalid here (approx exceeds 1, a nonsensical correlation value) even
    # though the exact formula stays properly bounded.
    exact = exact_current_correlation(p=0.2, r_in=0.025, n=250)
    approx = approx_meq1_current_correlation(p=0.2, r_in=0.025, n=250)
    assert exact <= 1.0
    assert approx > 1.0

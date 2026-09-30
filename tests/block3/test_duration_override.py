"""Seam: parse_length_s -- the `--seconds N` smoke-test override shared by block3's two
entry points. A malformed invocation must RAISE, not silently fall back to the paper-scale
duration: the whole point of the flag is to run a 5-second check before committing a Slurm
array to runs of 27 hours each, and a silent fallback turns that check into the very run it
was meant to de-risk.
"""
import pytest

from block3.full_pass import parse_length_s


def test_the_override_is_used_when_given():
    assert parse_length_s(["prog", "0", "--seconds", "5"], default=200.0) == 5.0


def test_the_default_is_used_when_no_override_is_given():
    assert parse_length_s(["prog", "0"], default=200.0) == 200.0


def test_a_flag_with_no_value_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="--seconds"):
        parse_length_s(["prog", "0", "--seconds"], default=200.0)


def test_a_mistyped_flag_raises_instead_of_falling_back():
    for typo in ("--second", "--secs", "-seconds", "seconds"):
        with pytest.raises(ValueError, match="--seconds"):
            parse_length_s(["prog", "0", typo, "5"], default=200.0)


def test_a_non_numeric_value_raises():
    with pytest.raises(ValueError):
        parse_length_s(["prog", "0", "--seconds", "five"], default=200.0)


def test_transposed_tokens_raise():
    with pytest.raises(ValueError, match="--seconds"):
        parse_length_s(["prog", "0", "5", "--seconds"], default=200.0)

"""Seams: lsm.reservoir.bin_counts (liquid state layout) and lsm.readout.train_and_score."""
import numpy as np

from lsm.readout import train_and_score
from lsm.reservoir import bin_counts


def test_spikes_land_in_their_sample_neuron_and_bin_and_gaps_are_ignored():
    onsets = np.array([100.0, 500.0])          # window 300 ms, 3 bins of 100 ms
    neuron = np.array([0, 1, 1, 0, 1])
    t = np.array([150.0, 399.0, 450.0, 510.0, 50.0])   # 450 is in the gap, 50 in burn-in
    f = bin_counts(neuron, t, n_neurons=2, onsets=onsets, window_ms=300.0, n_bins=3)
    assert f.shape == (2, 6)
    assert list(f[0]) == [1, 0, 0, 0, 0, 1]    # neuron 0 bin 0; neuron 1 bin 2
    assert list(f[1]) == [1, 0, 0, 0, 0, 0]


def test_readout_scaler_is_fit_on_training_data_only():
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(200, 3)); train_y = (train_x[:, 0] > 0).astype(int)
    test_x = train_x + 1000.0                  # shifted test set: a leaky scaler would hide it
    score = train_and_score(train_x, train_y, test_x, train_y, c=1.0, max_iter=500)
    assert score["train_accuracy"] > 0.9
    assert score["test_accuracy"] < 0.7


def test_regularisation_is_chosen_on_a_validation_split_of_the_training_data():
    rng = np.random.default_rng(1)
    train_x = rng.normal(size=(300, 20)); train_y = (train_x[:, 0] > 0).astype(int)
    test_x = rng.normal(size=(100, 20)); test_y = (test_x[:, 0] > 0).astype(int)
    score = train_and_score(train_x, train_y, test_x, test_y, c=[1e-4, 1.0], max_iter=500,
                            validation_fraction=0.25, rng=rng)
    assert score["C"] == 1.0                  # 1e-4 under-fits a clean linear problem
    assert score["test_accuracy"] > 0.9


def test_participation_ratio_counts_effective_dimensions():
    from lsm.readout import participation_ratio

    rng = np.random.default_rng(2)
    isotropic = rng.normal(size=(20000, 10))
    assert abs(participation_ratio(isotropic) - 10) < 0.3
    rank_one = np.outer(rng.normal(size=5000), rng.normal(size=8))
    assert abs(participation_ratio(rank_one) - 1) < 1e-6

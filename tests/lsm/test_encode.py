"""Seam: lsm.encode -- event stream -> X-population spike train. Truth: hand-worked events."""
import numpy as np

from lsm.encode import keep_probability, to_channels


def test_events_pool_onto_polarity_split_channels_and_leave_the_window():
    # 4x4 sensor pooled by 2 -> 2x2 sites x 2 polarities = 8 channels
    x = np.array([0, 1, 3, 2, 0]); y = np.array([0, 1, 0, 3, 0])
    p = np.array([0, 1, 1, 0, 0]); t = np.array([0.0, 1.0, 2.0, 3.0, 99.0])
    channel, t_out = to_channels(x, y, p, t, sensor=(4, 4), pool=2, window_ms=10.0)
    # (0,0,p0)->site 0 ch 0; (1,1,p1)->site 0 ch 1; (3,0,p1)->site 1 ch 3; (2,3,p0)->site 3 ch 6
    assert list(channel) == [0, 1, 3, 6] and list(t_out) == [0.0, 1.0, 2.0, 3.0]


def test_keep_probability_targets_the_mean_channel_rate_and_never_exceeds_one():
    # 8 channels, 100 ms window, 20 Hz target -> 16 events wanted per sample
    assert np.isclose(keep_probability(mean_events=64, n_channels=8, window_ms=100.0, target_hz=20.0), 0.25)
    assert keep_probability(mean_events=4, n_channels=8, window_ms=100.0, target_hz=20.0) == 1.0

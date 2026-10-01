import numpy as np
import pytest

pytest.importorskip("keras")

from anomaly_detector.model import build_conv_ae


def test_builds_with_expected_shapes():
    m = build_conv_ae(window_size=32, n_channels=8, latent_dim=16, filter_count=16)
    assert m.input_shape == (None, 32, 8)
    assert m.output_shape == (None, 32, 8)
    assert m.get_layer("latent").output.shape[-1] == 16


def test_latent_dim_and_filters_change_size():
    small = build_conv_ae(32, 8, latent_dim=4, filter_count=8).count_params()
    big = build_conv_ae(32, 8, latent_dim=32, filter_count=32).count_params()
    assert small < big


def test_invalid_window_size():
    with pytest.raises(ValueError, match="divisible by 4"):
        build_conv_ae(30, 8)


def test_trains_on_small_fixture():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 32, 8)).astype("float32")
    m = build_conv_ae(32, 8, latent_dim=4, filter_count=4)
    m.compile(optimizer="adam", loss="mse")
    hist = m.fit(x, x, epochs=3, batch_size=16, verbose=0)
    assert np.isfinite(hist.history["loss"]).all()
    assert m.predict(x[:5], verbose=0).shape == (5, 32, 8)

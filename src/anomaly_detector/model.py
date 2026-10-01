"""Small Conv1D autoencoder.

Input (W, C) -> Conv1D -> pool -> Conv1D -> pool -> Dense(latent_dim)   [encoder]
             -> Dense -> reshape -> upsample -> Conv1D -> upsample -> Conv1D(C)  [decoder]
"""
from __future__ import annotations

import keras
from keras import layers


def build_conv_ae(
    window_size: int,
    n_channels: int,
    latent_dim: int = 16,
    filter_count: int = 16,
    kernel_size: int = 5,
) -> keras.Model:
    """Build the (uncompiled) autoencoder. window_size must be divisible by 4."""
    if window_size % 4 != 0:
        raise ValueError(f"window_size must be divisible by 4, got {window_size}")
    if min(n_channels, latent_dim, filter_count) < 1 or kernel_size < 1:
        raise ValueError("n_channels, latent_dim, filter_count, kernel_size must be >= 1")
    t = window_size // 4  # temporal length at the bottleneck

    inp = keras.Input(shape=(window_size, n_channels), name="window")
    x = layers.Conv1D(filter_count, kernel_size, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(filter_count, kernel_size, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    x = layers.Dense(t * filter_count, activation="relu")(latent)
    x = layers.Reshape((t, filter_count))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filter_count, kernel_size, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(n_channels, kernel_size, padding="same", name="reconstruction")(x)
    return keras.Model(inp, out, name="conv_ae")

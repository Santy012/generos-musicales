"""Prueba de humo: el paquete importa y TensorFlow, librosa y ffmpeg funcionan juntos."""

import subprocess

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

import generos


def test_paquete_importa():
    assert generos.RANDOM_STATE == 42
    assert generos.SAMPLE_RATE == 22_050


def test_ffmpeg_convierte_y_librosa_extrae_mfcc(tmp_path):
    # Tono estéreo a 44,1 kHz; ffmpeg lo lleva al formato GTZAN (mono, 22.050 Hz).
    t = np.linspace(0, 5, 5 * 44_100, endpoint=False)
    tono = 0.5 * np.sin(2 * np.pi * 440 * t)
    origen = tmp_path / "origen.wav"
    destino = tmp_path / "destino.wav"
    sf.write(origen, np.column_stack([tono, tono]), 44_100)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(origen),
         "-ac", "1", "-ar", str(generos.SAMPLE_RATE), str(destino)],
        check=True,
    )

    y, sr = librosa.load(destino, sr=None)
    assert sr == generos.SAMPLE_RATE
    assert y.ndim == 1 and abs(len(y) - 5 * sr) < sr // 100

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    assert mfcc.shape[0] == 13 and np.isfinite(mfcc).all()


def test_densa_y_lstm_entrenan():
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 216, 13)).astype("float32")  # 5 s de MFCC con hop 512
    y = rng.integers(0, 10, size=32)

    densa = tf.keras.Sequential([
        tf.keras.Input(shape=X.shape[1:]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    lstm = tf.keras.Sequential([
        tf.keras.Input(shape=X.shape[1:]),
        tf.keras.layers.LSTM(8),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    for modelo in (densa, lstm):
        modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        historia = modelo.fit(X, y, epochs=1, batch_size=16, verbose=0)
        assert np.isfinite(historia.history["loss"][0])
        assert modelo.predict(X, verbose=0).shape == (32, 10)

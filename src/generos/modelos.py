"""Red densa y LSTM sobre la misma entrada: un segmento de MFCC (frames × coeficientes)."""

import tensorflow as tf

from generos.mfcc import FRAMES
from generos.config import N_MFCC

ENTRADA = (FRAMES, N_MFCC)
L2 = 1e-4


def densa(n_clases: int, dropout: float) -> tf.keras.Model:
    """Perceptrón 128-64 sobre el segmento aplanado, con L2 y dropout."""
    reg = tf.keras.regularizers.L2(L2)
    return tf.keras.Sequential([
        tf.keras.Input(shape=ENTRADA),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_clases, activation="softmax"),
    ], name="densa")


def lstm(n_clases: int, dropout: float) -> tf.keras.Model:
    """Dos capas LSTM (128 y 64) que recorren el segmento en el tiempo."""
    return tf.keras.Sequential([
        tf.keras.Input(shape=ENTRADA),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_clases, activation="softmax"),
    ], name="lstm")


REGISTRO = {"densa": densa, "lstm": lstm}


def construir(nombre: str, n_clases: int, dropout: float) -> tf.keras.Model:
    return REGISTRO[nombre](n_clases, dropout)


def representacion(modelo: tf.keras.Model) -> tf.keras.Model:
    """Submodelo que devuelve la penúltima capa densa (la representación aprendida)."""
    capa = [c for c in modelo.layers if isinstance(c, tf.keras.layers.Dense)][-2]
    return tf.keras.Model(modelo.inputs, capa.output)

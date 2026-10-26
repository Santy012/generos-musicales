"""Entrenamiento con validación separada y parada temprana; nunca mira la prueba."""

import time

import numpy as np
import tensorflow as tf

from generos import evaluacion, modelos
from generos.datos import Conjunto

EPOCAS_MAX = 150
PACIENCIA = 15
LOTE = 64


def entrenar(c: Conjunto, modelo: str, dropout: float, tasa: float, semilla: int):
    """Entrena sobre `train`, elige la mejor época por pérdida en `valid`."""
    tf.keras.utils.set_random_seed(semilla)
    red = modelos.construir(modelo, len(c.etiquetas.generos), dropout)
    red.compile(optimizer=tf.keras.optimizers.Adam(tasa), loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    parada = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
    inicio = time.perf_counter()
    historia = red.fit(c.train.X, c.train.y, validation_data=(c.valid.X, c.valid.y),
                       epochs=EPOCAS_MAX, batch_size=LOTE, callbacks=[parada], verbose=0)
    segundos = time.perf_counter() - inicio
    return red, historia.history, segundos


def experimento(c: Conjunto, modelo: str, dropout: float, tasa: float, semilla: int, evaluar_prueba: bool):
    """Entrena y evalúa. La prueba solo se evalúa en las corridas finales."""
    red, hist, segundos = entrenar(c, modelo, dropout, tasa, semilla)
    n = len(c.etiquetas.generos)
    mejor = int(np.argmin(hist["val_loss"]))
    resultado = {
        "escenario": c.escenario, "modelo": modelo, "dropout": dropout, "tasa": tasa, "semilla": semilla,
        "generos": list(c.etiquetas.generos), "parametros": int(red.count_params()),
        "epocas": len(hist["loss"]), "mejor_epoca": mejor + 1, "segundos": round(segundos, 1),
        "historia": {k: [round(float(v), 5) for v in vs] for k, vs in hist.items()},
        "valid": _sin_detalle(evaluacion.metricas(red.predict(c.valid.X, verbose=0), c.valid.y, c.valid.cancion, n)),
    }
    if evaluar_prueba:
        resultado["test"] = evaluacion.metricas(red.predict(c.test.X, verbose=0), c.test.y, c.test.cancion, n)
    return red, resultado


def _sin_detalle(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("predicciones_cancion", "confusion_segmento")}

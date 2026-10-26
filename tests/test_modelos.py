"""Conteo de parámetros fijado y voto por canción."""

import numpy as np
import pytest

from generos import evaluacion, modelos


@pytest.mark.parametrize("nombre, n_clases, esperado", [
    ("densa", 10, 2808 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10),
    ("lstm", 10, 4 * (128 * (13 + 128) + 128) + 4 * (64 * (128 + 64) + 64) + 64 * 64 + 64 + 64 * 10 + 10),
    ("densa", 14, 2808 * 128 + 128 + 128 * 64 + 64 + 64 * 14 + 14),
])
def test_parametros(nombre, n_clases, esperado):
    assert modelos.construir(nombre, n_clases, 0.3).count_params() == esperado


def test_representacion_es_la_penultima_capa():
    red = modelos.construir("lstm", 12, 0.3)
    assert modelos.representacion(red).output.shape[-1] == 64


def test_voto_mayoria_y_desempate():
    prob = np.array([
        [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.1, 0.9, 0.0],   # canción a: gana 0 por mayoría
        [0.6, 0.4, 0.0], [0.1, 0.9, 0.0],                    # canción b: empate, desempata 1
    ])
    canciones = np.array(["a", "a", "a", "b", "b"])
    v = evaluacion.voto(prob, canciones).set_index("cancion")["pred"]
    assert v["a"] == 0 and v["b"] == 1


def test_metricas_por_cancion():
    prob = np.eye(3)[[0, 0, 1, 2, 2, 2]]
    y = np.array([0, 0, 1, 1, 2, 2])
    canciones = np.array(["a", "a", "b", "b", "c", "c"])
    m = evaluacion.metricas(prob, y, canciones, 3)
    assert m["exactitud_segmento"] == pytest.approx(5 / 6)
    assert m["n_canciones"] == 3
    assert np.trace(np.array(m["confusion_cancion"])) == round(m["exactitud_cancion"] * 3)

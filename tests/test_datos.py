"""Etiquetas, segmentos y particiones: el mapeo índice → género y la ausencia de fugas."""

import numpy as np
import pandas as pd
import pytest

from generos import datos, extension
from generos.config import ESCENARIOS, GENEROS_GTZAN, GENEROS_NUEVOS
from generos.mfcc import CACHE, FRAMES

hay_mfcc = pytest.mark.skipif(not CACHE.exists(), reason="requiere los MFCC en data/ (scripts/run.py)")
hay_extension = pytest.mark.skipif(not extension.USADAS.exists(), reason="requiere scripts/descargar_extension.py")


def test_etiquetas_ida_y_vuelta():
    e = datos.Etiquetas.de("C")
    assert e.nombres(e.indices(e.generos)) == list(e.generos)


def test_escenarios_amplian_sin_mover_indices():
    a, b, c = (datos.Etiquetas.de(x) for x in "ABC")
    assert a.generos == GENEROS_GTZAN
    assert b.generos[:10] == a.generos and c.generos[:12] == b.generos
    assert set(c.generos) == set(GENEROS_GTZAN) | set(GENEROS_NUEVOS)
    assert all(len(set(g)) == len(g) for g in ESCENARIOS.values())


def test_particion_por_artista_sintetica():
    df = pd.DataFrame({"grupo": [f"a{i // 3}" for i in range(99)]})  # 33 artistas de 3 canciones
    p = extension.particion_por_artista(df)
    assert (df.assign(p=p).groupby("grupo")["p"].nunique() == 1).all()
    proporcion = p.value_counts(normalize=True)
    assert abs(proporcion["train"] - 0.70) < 0.05
    assert abs(proporcion["valid"] - 0.15) < 0.05 and abs(proporcion["test"] - 0.15) < 0.05


def test_alias_unen_grupo_y_solista():
    assert extension.grupo_artista("Gustavo Cerati") == extension.grupo_artista("Soda Stereo")
    assert extension.grupo_artista("Los Rodríguez") == extension.grupo_artista("Andrés Calamaro")


@hay_extension
def test_extension_sin_artistas_compartidos():
    usadas = pd.read_csv(extension.USADAS)
    assert (usadas.groupby("grupo")["particion"].nunique() == 1).all()
    assert usadas["id_video"].is_unique
    for genero, g in usadas.groupby("genero"):
        assert len(g) >= extension.MINIMO
        assert set(g["particion"]) == {"train", "valid", "test"}


@hay_mfcc
@pytest.mark.parametrize("escenario", ["A"])
def test_conjunto_etiquetas_y_fugas(escenario):
    c = datos.conjunto(escenario)
    genero_de = c.canciones.set_index("cancion")["genero"]
    for p in (c.train, c.valid, c.test):
        assert p.X.shape[1:] == (FRAMES, 13)
        # la etiqueta de cada segmento es el género de su canción en el catálogo
        assert c.etiquetas.nombres(p.y) == list(genero_de.loc[p.cancion])
    canciones = [set(p.cancion) for p in (c.train, c.valid, c.test)]
    assert not (canciones[0] & canciones[1] or canciones[0] & canciones[2] or canciones[1] & canciones[2])
    # normalización con estadísticas del entrenamiento
    assert np.allclose(c.train.X.mean(axis=(0, 1)), 0, atol=1e-3)
    assert np.allclose(c.train.X.std(axis=(0, 1)), 1, atol=1e-3)

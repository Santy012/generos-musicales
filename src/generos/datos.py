"""Catálogo de canciones, etiquetas y conjuntos de segmentos por escenario."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from generos import extension, gtzan, mfcc
from generos.config import ESCENARIOS


@dataclass(frozen=True)
class Etiquetas:
    """Mapeo índice ↔ género de un escenario. Es el único lugar donde se fija."""

    generos: tuple[str, ...]

    @classmethod
    def de(cls, escenario: str) -> "Etiquetas":
        return cls(ESCENARIOS[escenario])

    def indices(self, generos) -> np.ndarray:
        posicion = {g: i for i, g in enumerate(self.generos)}
        return np.array([posicion[g] for g in generos], dtype=np.int64)

    def nombres(self, indices) -> list[str]:
        return [self.generos[i] for i in indices]


@dataclass
class Particion:
    X: np.ndarray        # (n_segmentos, frames, n_mfcc), normalizado
    y: np.ndarray        # índice de género por segmento
    cancion: np.ndarray  # id de la canción de cada segmento


@dataclass
class Conjunto:
    escenario: str
    etiquetas: Etiquetas
    train: Particion
    valid: Particion
    test: Particion
    canciones: pd.DataFrame
    media: np.ndarray
    desviacion: np.ndarray


def catalogo() -> pd.DataFrame:
    """Todas las canciones: GTZAN y, si ya se descargaron, los géneros nuevos."""
    partes = [gtzan.particion()]
    if extension.USADAS.exists():
        partes.append(extension.particion())
    return pd.concat(partes, ignore_index=True)


def conjunto(escenario: str, cat: pd.DataFrame | None = None) -> Conjunto:
    """Segmentos del escenario, normalizados con media y desviación del entrenamiento."""
    etiquetas = Etiquetas.de(escenario)
    cat = catalogo() if cat is None else cat
    cat = cat[cat["genero"].isin(etiquetas.generos)].reset_index(drop=True)
    por_cancion = mfcc.extraer(cat)

    def armar(nombre: str) -> Particion:
        filas = cat[cat["particion"] == nombre]
        X = np.concatenate([por_cancion[c] for c in filas["cancion"]])
        n = [len(por_cancion[c]) for c in filas["cancion"]]
        return Particion(X=X, y=np.repeat(etiquetas.indices(filas["genero"]), n),
                         cancion=np.repeat(filas["cancion"].to_numpy(), n))

    train, valid, test = armar("train"), armar("valid"), armar("test")
    media = train.X.mean(axis=(0, 1))
    desviacion = train.X.std(axis=(0, 1))
    for p in (train, valid, test):
        p.X = ((p.X - media) / desviacion).astype(np.float32)
    return Conjunto(escenario, etiquetas, train, valid, test, cat, media, desviacion)

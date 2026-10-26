"""Géneros nuevos: listas curadas, canciones usadas y partición por artista.

Las listas viven en `data_sources/<genero>.csv` (una fila por canción candidata, con
el video de YouTube, el segundo de inicio y la columna `validado`). El audio se
descarga con `scripts/descargar_extension.py` a `data/extension/` y nunca se versiona.
Las canciones finalmente usadas y su partición quedan fijadas en
`data_sources/extension_usadas.csv`.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from generos import RANDOM_STATE
from generos.config import DATA, DATA_SOURCES, GENEROS_NUEVOS

DIR = DATA / "extension"
USADAS = DATA_SOURCES / "extension_usadas.csv"
POR_GENERO = 100
MINIMO = 60
PROPORCIONES = {"train": 0.70, "valid": 0.15, "test": 0.15}
_VACIAS = {"los", "las", "el", "la", "y", "sus", "de", "del", "orquesta", "grupo", "hermanos"}


def clave_artista(artista: str) -> str:
    """Normaliza el artista principal: sin tildes, artículos ni colaboradores."""
    s = unicodedata.normalize("NFKD", artista.split("&")[0]).encode("ascii", "ignore").decode().lower()
    return " ".join(p for p in re.sub(r"[^a-z0-9 ]+", " ", s).split() if p not in _VACIAS)


# Artistas que comparten integrantes, voz o discos: cuentan como un solo grupo en la
# partición para que el modelo no vea al mismo intérprete en entrenamiento y prueba.
_ALIAS = [
    ("Willie Colón", "Héctor Lavoe", "Rubén Blades"),
    ("Fruko y Sus Tesos", "Joe Arroyo"),
    ("El Gran Combo de Puerto Rico", "Andy Montañez"),
    ("Hermanos Zuleta", "Poncho Zuleta", "Emilianito Zuleta", "Emiliano Zuleta", "Iván Zuleta"),
    ("Binomio de Oro", "Rafael Orozco"),
    ("Los Gigantes del Vallenato", "Daniel Calderón"),
    ("Los Betos", "Beto Zabaleta"),
    ("Kaleth Morales", "Los K Morales"),
    ("Wisin & Yandel", "Wisin", "Yandel"),
    ("Plan B", "Chencho Corleone"),
    ("Soda Stereo", "Gustavo Cerati"),
    ("Héroes del Silencio", "Enrique Bunbury"),
    ("Caifanes", "Jaguares"),
    ("Andrés Calamaro", "Los Rodríguez", "Los Abuelos de la Nada"),
    ("Patricio Rey y sus Redonditos de Ricota", "Indio Solari"),
    ("Sumo", "Divididos"),
]
ALIAS = {clave_artista(a): clave_artista(grupo[0]) for grupo in _ALIAS for a in grupo}


def grupo_artista(artista: str) -> str:
    clave = clave_artista(artista)
    return ALIAS.get(clave, clave)


def candidatas(genero: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_SOURCES / f"{genero}.csv", dtype={"id_video": str})
    df["grupo"] = genero + ":" + df["artista"].map(grupo_artista)
    return df


def validadas(genero: str) -> pd.DataFrame:
    df = candidatas(genero)
    return df[df["validado"].notna() & (df["validado"].astype(str).str.strip() != "")]


def ruta_audio(genero: str, id_video: str):
    return DIR / genero / f"{id_video}.wav"


def particion_por_artista(df: pd.DataFrame, semilla: int = RANDOM_STATE) -> pd.Series:
    """Asigna artistas completos a train/valid/test buscando 70/15/15 de canciones.

    Recorre los artistas en orden aleatorio fijo y pone cada uno en la partición que
    más lejos está de su cuota. Ningún artista queda en dos particiones.
    """
    rng = np.random.default_rng(semilla)
    conteo = df.groupby("grupo").size()
    artistas = conteo.index.to_numpy()[rng.permutation(len(conteo))]
    total = len(df)
    asignadas = dict.fromkeys(PROPORCIONES, 0)
    destino = {}
    for a in artistas:
        deficit = {p: PROPORCIONES[p] * total - asignadas[p] for p in PROPORCIONES}
        p = max(deficit, key=deficit.get)
        destino[a] = p
        asignadas[p] += conteo[a]
    return df["grupo"].map(destino)


def particion() -> pd.DataFrame:
    """Una fila por canción usada: cancion, genero, grupo, fuente, particion, ruta."""
    df = pd.read_csv(USADAS, dtype={"id_video": str})
    df["cancion"] = df["genero"] + "." + df["id_video"]
    df["fuente"] = "youtube"
    df["ruta"] = [ruta_audio(g, i) for g, i in zip(df["genero"], df["id_video"])]
    return df[["cancion", "genero", "grupo", "fuente", "particion", "ruta", "artista", "titulo"]]


def fijar_usadas(descargadas: dict[str, list[str]]) -> pd.DataFrame:
    """Guarda las canciones usadas (por género, en orden de la lista) y su partición."""
    partes = []
    for genero in GENEROS_NUEVOS:
        ids = descargadas.get(genero, [])
        if len(ids) < MINIMO:
            raise RuntimeError(f"{genero}: solo {len(ids)} canciones; el mínimo es {MINIMO}.")
        df = validadas(genero).set_index("id_video").loc[ids].reset_index()
        df["particion"] = particion_por_artista(df).to_numpy()
        partes.append(df[["genero", "id_video", "artista", "titulo", "grupo", "particion"]])
    usadas = pd.concat(partes, ignore_index=True)
    usadas.to_csv(USADAS, index=False, lineterminator="\n")
    return usadas

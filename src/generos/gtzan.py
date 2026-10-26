"""GTZAN: descarga verificada y partición fault-filtered.

La partición es la de Kereliuk, Sturm y Larsen (2015), publicada en el repositorio
`coreyker/dnn-mgr`: descarta las repeticiones y distorsiones conocidas de GTZAN y no
comparte artistas entre entrenamiento, validación y prueba.
"""

import shutil
from pathlib import Path

import pandas as pd

from generos.config import DATA, DATA_SOURCES, GENEROS_GTZAN
from generos.verificacion import descargar as descargar_url, sha256

KAGGLE = "andradaolteanu/gtzan-dataset-music-genre-classification"
DIR = DATA / "gtzan"
EXCLUIDAS = {"jazz.00054"}  # archivo dañado: librosa no lo puede leer

_COMMIT = "bdad579ea6cb37b665ea6019fe1026a6ce20cbc7"
_URL = f"https://raw.githubusercontent.com/coreyker/dnn-mgr/{_COMMIT}/gtzan/{{}}_filtered.txt"
LISTAS = {
    "train": "2b04966cef1486f1a7767b74271533fa23e220388db804a1937835e1fc4fa3fa",
    "valid": "f8fbee7c52397a0e0f3a91dc54fedecb2e27f5bdc9a9daeab8cb10d077e26956",
    "test": "09cafa6a89ce0a27f30ed250d56ecb4d03c22fffb5a693f96c7ae655ee02ac72",
}
HASHES = DATA_SOURCES / "gtzan_sha256.csv"


def particion() -> pd.DataFrame:
    """Una fila por canción: cancion, genero, grupo, fuente, particion, ruta."""
    filas = []
    for nombre, sha in LISTAS.items():
        lista = descargar_url(_URL.format(nombre), DIR / "splits" / f"{nombre}_filtered.txt", sha)
        for linea in lista.read_text().split():
            genero, archivo = linea.split("/")
            cancion = archivo.removesuffix(".wav")
            if cancion in EXCLUIDAS:
                continue
            filas.append({"cancion": cancion, "genero": genero, "grupo": f"gtzan:{cancion}",
                          "fuente": "gtzan", "particion": nombre,
                          "ruta": DIR / "audio" / genero / archivo})
    df = pd.DataFrame(filas)
    assert set(df["genero"]) == set(GENEROS_GTZAN)
    return df


def descargar() -> pd.DataFrame:
    """Copia los WAV usados desde la caché de kagglehub y verifica su SHA-256."""
    df = particion()
    faltan = df[~df["ruta"].map(lambda r: r.exists())]
    if len(faltan):
        import kagglehub

        origen = kagglehub.dataset_download(KAGGLE)
        base = next(p for p in Path(origen).rglob("genres_original") if p.is_dir())
        for r in faltan.itertuples():
            r.ruta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / r.genero / r.ruta.name, r.ruta)
    verificar(df)
    return df


def verificar(df: pd.DataFrame) -> None:
    """Exige el SHA-256 registrado en data_sources/. La primera vez lo registra."""
    obtenidos = {r.cancion: sha256(r.ruta) for r in df.itertuples()}
    if not HASHES.exists():
        pd.DataFrame(sorted(obtenidos.items()), columns=["cancion", "sha256"]).to_csv(
            HASHES, index=False, lineterminator="\n")
        return
    esperados = pd.read_csv(HASHES).set_index("cancion")["sha256"].to_dict()
    distintos = [c for c, h in obtenidos.items() if esperados.get(c) != h]
    if distintos:
        raise ValueError(f"SHA-256 distinto o no registrado en {len(distintos)} archivos: {distintos[:5]}")

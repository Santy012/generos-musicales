"""MFCC de 13 coeficientes sobre segmentos de 5 s, con caché en data/mfcc/."""

from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd

from generos.config import DATA, DURACION_CANCION, DURACION_SEGMENTO, HOP_LENGTH, N_FFT, N_MFCC, SAMPLE_RATE

CACHE = DATA / "mfcc" / "segmentos.npz"
MUESTRAS_SEGMENTO = SAMPLE_RATE * DURACION_SEGMENTO
SEGMENTOS_POR_CANCION = DURACION_CANCION // DURACION_SEGMENTO
FRAMES = 1 + MUESTRAS_SEGMENTO // HOP_LENGTH  # 216 con center=True


def segmentos(ruta) -> np.ndarray:
    """(n_segmentos, FRAMES, N_MFCC) de una canción. Descarta el resto de menos de 5 s."""
    y, _ = librosa.load(ruta, sr=SAMPLE_RATE, mono=True)
    n = min(len(y) // MUESTRAS_SEGMENTO, SEGMENTOS_POR_CANCION)
    salida = np.empty((n, FRAMES, N_MFCC), dtype=np.float32)
    for i in range(n):
        trozo = y[i * MUESTRAS_SEGMENTO:(i + 1) * MUESTRAS_SEGMENTO]
        m = librosa.feature.mfcc(y=trozo, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        salida[i] = m.T
    return salida


def extraer(catalogo: pd.DataFrame, procesos: int = 6) -> dict[str, np.ndarray]:
    """MFCC de todas las canciones del catálogo; reutiliza la caché y la amplía si faltan."""
    previos = _leer_cache()
    faltan = [c for c in catalogo["cancion"] if c not in previos]
    if faltan:
        rutas = catalogo.set_index("cancion").loc[faltan, "ruta"]
        with ProcessPoolExecutor(procesos) as ex:
            for c, x in zip(faltan, ex.map(segmentos, rutas, chunksize=8)):
                previos[c] = x
        _escribir_cache(previos)
    return {c: previos[c] for c in catalogo["cancion"]}


def _leer_cache() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        return {}
    with np.load(CACHE) as z:
        X, canciones, inicio = z["X"], z["canciones"], z["inicio"]
    return {c: X[inicio[k]:inicio[k + 1]] for k, c in enumerate(canciones)}


def _escribir_cache(por_cancion: dict[str, np.ndarray]) -> None:
    canciones = np.array(sorted(por_cancion))
    bloques = [por_cancion[c] for c in canciones]
    inicio = np.concatenate([[0], np.cumsum([len(b) for b in bloques])])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_name("segmentos.tmp.npz")
    np.savez(tmp, X=np.concatenate(bloques), canciones=canciones, inicio=inicio)
    tmp.replace(CACHE)

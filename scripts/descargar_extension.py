"""Descarga el audio de los géneros nuevos a partir de las listas de data_sources/.

Por cada género recorre las canciones validadas en orden y se queda con las primeras
100 que se pueden descargar. Cada una se recorta a 30 s desde su segundo de inicio y
se convierte al formato de GTZAN (WAV mono, 22.050 Hz). El audio queda en
data/extension/ y no se versiona. Los fallos van a logs/descargas.log.

Al terminar fija las canciones usadas y su partición por artista en
data_sources/extension_usadas.csv. Si ese archivo ya existe, descarga solo esas.

Uso:
    python scripts/descargar_extension.py
"""

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import pandas as pd
import yt_dlp

from generos.config import DURACION_CANCION, GENEROS_NUEVOS, LOGS, SAMPLE_RATE
from generos.extension import DIR, POR_GENERO, USADAS, fijar_usadas, ruta_audio, validadas

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
log = logging.getLogger("descargas")


def descargar(genero: str, id_video: str, inicio: int) -> bool:
    destino = ruta_audio(genero, id_video)
    if destino.exists():
        return True
    tmp = DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    opciones = {"format": "bestaudio/best", "outtmpl": str(tmp / "%(id)s.%(ext)s"),
                "quiet": True, "no_warnings": True, "noprogress": True}
    try:
        with yt_dlp.YoutubeDL(opciones) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={id_video}", download=True)
            original = ydl.prepare_filename(info)
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_suffix(".tmp.wav")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", str(inicio), "-t", str(DURACION_CANCION),
                        "-i", original, "-ac", "1", "-ar", str(SAMPLE_RATE), str(temporal)],
                       check=True, capture_output=True)
        temporal.replace(destino)
        return True
    except Exception as e:  # video borrado, restringido, error de red o de ffmpeg
        log.warning("%s %s: %s", genero, id_video, str(e).splitlines()[0][:200])
        return False
    finally:
        for f in tmp.glob(f"{id_video}.*"):
            f.unlink(missing_ok=True)


def descargar_genero(genero: str, filas: pd.DataFrame, objetivo: int) -> list[str]:
    """Descarga en lotes hasta tener `objetivo` canciones; devuelve sus ids en orden de la lista."""
    ok, pos = [], 0
    while len(ok) < objetivo and pos < len(filas):
        lote = filas.iloc[pos:pos + objetivo - len(ok)]
        pos += len(lote)
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda r: descargar(genero, r.id_video, int(r.inicio)), lote.itertuples()))
        ok += [r.id_video for r, bien in zip(lote.itertuples(), res) if bien]
    orden = {i: k for k, i in enumerate(filas["id_video"])}
    return sorted(ok, key=orden.get)


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    logging.basicConfig(filename=LOGS / "descargas.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", encoding="utf-8")
    if USADAS.exists():
        usadas = pd.read_csv(USADAS, dtype={"id_video": str})
        for genero, g in usadas.groupby("genero"):
            filas = validadas(genero).set_index("id_video").loc[g["id_video"]].reset_index()
            ok = descargar_genero(genero, filas, len(filas))
            print(f"{genero}: {len(ok)} de {len(filas)} canciones fijadas")
        return
    descargadas = {}
    for genero in GENEROS_NUEVOS:
        descargadas[genero] = descargar_genero(genero, validadas(genero), POR_GENERO)
        print(f"{genero}: {len(descargadas[genero])} canciones")
    usadas = fijar_usadas(descargadas)
    print(usadas.groupby(["genero", "particion"]).size().unstack())


if __name__ == "__main__":
    main()

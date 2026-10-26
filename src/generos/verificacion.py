"""Descargas con verificación SHA-256."""

import hashlib
import urllib.request
from pathlib import Path


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url: str, destino: Path, sha_esperado: str) -> Path:
    """Descarga `url` a `destino` si no está y exige el SHA-256 indicado."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        tmp = destino.with_suffix(destino.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(destino)
    obtenido = sha256(destino)
    if obtenido != sha_esperado.lower():
        raise ValueError(f"SHA-256 distinto en {destino.name}: {obtenido}")
    return destino

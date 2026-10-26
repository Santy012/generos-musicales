"""Corre los experimentos de configs/ y guarda un JSON por corrida en results/.

Reanudable: salta las corridas que ya tienen resultado. Un error queda en
logs/run.log y la corrida sigue con la siguiente.

Etapas:
    busqueda   rejilla de hiperparámetros en validación (escenario A, semilla 0)
    principal  escenarios A, B y C, ambos modelos, tres semillas; evalúa la prueba

Uso:
    python scripts/run.py busqueda
    python scripts/run.py principal
    python scripts/run.py todo
"""

import argparse
import logging
import tomllib
import traceback

import numpy as np

from generos import datos, entrenamiento, evaluacion, mfcc
from generos import resultados as res
from generos.config import DATA, ESCENARIOS, GENEROS_NUEVOS, LOGS, RAIZ

MODELOS_GUARDADOS = DATA / "modelos"
log = logging.getLogger("run")
_conjuntos: dict[str, datos.Conjunto] = {}


def conjunto(escenario: str) -> datos.Conjunto:
    if escenario not in _conjuntos:
        _conjuntos[escenario] = datos.conjunto(escenario)
    return _conjuntos[escenario]


def correr(destino, escenario, modelo, dropout, tasa, semilla, evaluar_prueba, guardar=False):
    ruta = destino / f"{res.nombre(escenario, modelo, dropout, tasa, semilla)}.json"
    if ruta.exists():
        return
    print(f"→ {ruta.stem}", flush=True)
    try:
        c = conjunto(escenario)
        red, r = entrenamiento.experimento(c, modelo, dropout, tasa, semilla, evaluar_prueba)
        if evaluar_prueba:
            r["fuera_de_escenario"] = fuera_de_escenario(c, red)
        if guardar:
            MODELOS_GUARDADOS.mkdir(parents=True, exist_ok=True)
            red.save(MODELOS_GUARDADOS / f"{ruta.stem}.keras")
        res.escribir(ruta, r)
        print(f"  {r['epocas']} épocas, {r['segundos']} s, valid {r['valid']['exactitud_segmento']:.3f}", flush=True)
    except Exception:
        log.error("%s\n%s", ruta.stem, traceback.format_exc())
        print("  error (ver logs/run.log)", flush=True)


def fuera_de_escenario(c: datos.Conjunto, red) -> dict:
    """Predicciones del modelo sobre las canciones de géneros nuevos que no conoce.

    Responde, por ejemplo, a qué género de GTZAN asigna el modelo B el rock en español.
    """
    cat = datos.catalogo()
    ajenas = cat[cat["genero"].isin(set(GENEROS_NUEVOS) - set(c.etiquetas.generos))]
    salida = {}
    if ajenas.empty:
        return salida
    por_cancion = mfcc.extraer(ajenas)
    for genero, g in ajenas.groupby("genero"):
        X = np.concatenate([por_cancion[s] for s in g["cancion"]])
        X = ((X - c.media) / c.desviacion).astype(np.float32)
        canciones = np.repeat(g["cancion"].to_numpy(), [len(por_cancion[s]) for s in g["cancion"]])
        votos = evaluacion.voto(red.predict(X, verbose=0), canciones)
        conteo = np.bincount(votos["pred"], minlength=len(c.etiquetas.generos))
        salida[genero] = dict(zip(c.etiquetas.generos, conteo.tolist()))
    return salida


def busqueda() -> None:
    cfg = tomllib.loads((RAIZ / "configs" / "busqueda.toml").read_text(encoding="utf-8"))
    for modelo in cfg["modelos"]:
        for dropout in cfg["dropout"]:
            for tasa in cfg["tasa"]:
                correr(res.BUSQUEDA, cfg["escenario"], modelo, dropout, tasa, cfg["semilla"], False)


def principal() -> None:
    cfg = tomllib.loads((RAIZ / "configs" / "principal.toml").read_text(encoding="utf-8"))
    hp = res.elegidos()
    print("Hiperparámetros elegidos en validación:", hp, flush=True)
    for semilla in cfg["semillas"]:
        for escenario in cfg["escenarios"]:
            for modelo in cfg["modelos"]:
                correr(res.CORRIDAS, escenario, modelo, hp[modelo]["dropout"], hp[modelo]["tasa"],
                       semilla, True, guardar=semilla == cfg["semillas"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etapa", choices=["busqueda", "principal", "todo"])
    etapa = parser.parse_args().etapa
    LOGS.mkdir(exist_ok=True)
    logging.basicConfig(filename=LOGS / "run.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", encoding="utf-8")
    assert set(ESCENARIOS) == {"A", "B", "C"}
    if etapa in ("busqueda", "todo"):
        busqueda()
    if etapa in ("principal", "todo"):
        principal()


if __name__ == "__main__":
    main()

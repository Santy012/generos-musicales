"""Lectura y escritura de results/ y tablas derivadas."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from generos.config import ESCENARIOS, NOMBRES, RESULTS

BUSQUEDA = RESULTS / "busqueda"
CORRIDAS = RESULTS / "corridas"
MODELOS = {"densa": "Red densa", "lstm": "LSTM"}


def nombre(escenario: str, modelo: str, dropout: float, tasa: float, semilla: int) -> str:
    return f"{escenario}_{modelo}_d{dropout:g}_lr{tasa:g}_s{semilla}"


def escribir(ruta: Path, datos: dict) -> None:
    """Escritura atómica: un corte nunca deja un JSON a medias."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, ruta)


def cargar(carpeta: Path = CORRIDAS) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(carpeta.glob("*.json"))]


def elegidos() -> dict[str, dict]:
    """Hiperparámetros por modelo: los de mayor exactitud por segmento en validación."""
    df = pd.DataFrame([{**{k: r[k] for k in ("modelo", "dropout", "tasa")},
                        "valid": r["valid"]["exactitud_segmento"]} for r in cargar(BUSQUEDA)])
    mejor = df.loc[df.groupby("modelo")["valid"].idxmax()]
    return {r.modelo: {"dropout": r.dropout, "tasa": r.tasa} for r in mejor.itertuples()}


def tabla_busqueda() -> pd.DataFrame:
    filas = [{"Modelo": MODELOS[r["modelo"]], "Dropout": r["dropout"], "Tasa": r["tasa"],
              "Épocas": r["mejor_epoca"], "Validación (segmento)": r["valid"]["exactitud_segmento"],
              "Validación (canción)": r["valid"]["exactitud_cancion"]} for r in cargar(BUSQUEDA)]
    return pd.DataFrame(filas).sort_values(["Modelo", "Dropout", "Tasa"])


def resumen(corridas: list[dict] | None = None) -> pd.DataFrame:
    """Media y desviación entre semillas por escenario y modelo (exactitud en prueba)."""
    corridas = cargar() if corridas is None else corridas
    df = pd.DataFrame([{"escenario": r["escenario"], "modelo": r["modelo"], "semilla": r["semilla"],
                        "parametros": r["parametros"],
                        "segmento": r["test"]["exactitud_segmento"],
                        "cancion": r["test"]["exactitud_cancion"]} for r in corridas])
    return (df.groupby(["escenario", "modelo"])
              .agg(n=("semilla", "size"), parametros=("parametros", "first"),
                   segmento_media=("segmento", "mean"), segmento_de=("segmento", "std"),
                   cancion_media=("cancion", "mean"), cancion_de=("cancion", "std"))
              .reset_index())


def confusion_media(escenario: str, modelo: str, nivel: str = "cancion") -> tuple[np.ndarray, list[str]]:
    """Matriz de confusión sumada sobre semillas y normalizada por fila (sensibilidad)."""
    rs = [r for r in cargar() if r["escenario"] == escenario and r["modelo"] == modelo]
    cm = np.sum([np.array(r["test"][f"confusion_{nivel}"]) for r in rs], axis=0)
    return cm / cm.sum(1, keepdims=True), rs[0]["generos"]


def sensibilidad_por_genero(escenario: str) -> pd.DataFrame:
    """Exactitud por canción de cada género (media sobre semillas), por modelo."""
    filas = {}
    for modelo in MODELOS:
        cm, generos = confusion_media(escenario, modelo)
        filas[MODELOS[modelo]] = pd.Series(np.diag(cm), index=[NOMBRES[g] for g in generos])
    return pd.DataFrame(filas)


def rock_vs_rock_es(escenario: str = "C") -> pd.DataFrame:
    """De las canciones de rock y de rock en español, % predicho como cada una (media de semillas)."""
    filas = []
    for modelo in MODELOS:
        cm, generos = confusion_media(escenario, modelo)
        i, j = generos.index("rock"), generos.index("rock_es")
        for real, k in (("rock_es", j), ("rock", i)):
            filas.append({"modelo": modelo, "real": real, "como_rock_es": cm[k, j], "como_rock": cm[k, i],
                          "como_otro": 1 - cm[k, i] - cm[k, j]})
    return pd.DataFrame(filas)


def fuera_de_escenario(genero: str, escenario: str = "B") -> pd.DataFrame:
    """Reparto de las canciones de un género no visto entre los géneros del modelo (todas las semillas)."""
    filas = {}
    for modelo in MODELOS:
        rs = [r for r in cargar() if r["escenario"] == escenario and r["modelo"] == modelo]
        conteo = pd.DataFrame([r["fuera_de_escenario"][genero] for r in rs]).sum()
        filas[modelo] = conteo / conteo.sum()
    return pd.DataFrame(filas)


def misma_fuente(escenario: str) -> pd.DataFrame:
    """Por fuente real (GTZAN o YouTube): % de canciones predichas como un género de la misma fuente.

    Se calcula con las matrices de confusión por canción sumadas sobre semillas.
    """
    from generos.config import GENEROS_GTZAN

    filas = []
    for modelo in MODELOS:
        rs = [r for r in cargar() if r["escenario"] == escenario and r["modelo"] == modelo]
        cm = np.sum([np.array(r["test"]["confusion_cancion"]) for r in rs], axis=0)
        generos = rs[0]["generos"]
        es_gtzan = np.array([g in GENEROS_GTZAN for g in generos])
        for fuente, filas_fuente in (("GTZAN", es_gtzan), ("YouTube", ~es_gtzan)):
            sub = cm[filas_fuente]
            filas.append({"modelo": modelo, "fuente": fuente,
                          "misma_fuente": sub[:, filas_fuente].sum() / sub.sum(),
                          "genero_correcto": np.trace(cm[np.ix_(filas_fuente, filas_fuente)]) / sub.sum()})
    return pd.DataFrame(filas)


def tabla_fuente(escenarios=("B", "C")) -> str:
    lineas = ["| Escenario | Modelo | Canciones de | Predichas como un género de la misma fuente (%) | Género correcto (%) |",
              "| --- | --- | --- | ---: | ---: |"]
    for e in escenarios:
        for r in misma_fuente(e).itertuples():
            lineas.append(f"| {e} | {MODELOS[r.modelo]} | {r.fuente} | {pct(r.misma_fuente)} | {pct(r.genero_correcto)} |")
    return "\n".join(lineas)


def tabla_rock() -> str:
    df = rock_vs_rock_es()
    lineas = ["| Modelo | Canciones de | Predichas como rock en español (%) | Predichas como rock (%) | Otro género (%) |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in df.itertuples():
        lineas.append(f"| {MODELOS[r.modelo]} | {NOMBRES[r.real]} | {pct(r.como_rock_es)} | {pct(r.como_rock)} "
                      f"| {pct(r.como_otro)} |")
    return "\n".join(lineas)


def tabla_fuera(generos=("rock_es", "reggaeton"), top: int = 3) -> str:
    lineas = ["| Género no visto | Modelo | Géneros asignados con más frecuencia (% de las canciones) |",
              "| --- | --- | --- |"]
    for g in generos:
        df = fuera_de_escenario(g)
        for modelo in MODELOS:
            s = df[modelo].sort_values(ascending=False).head(top)
            texto = ", ".join(f"{NOMBRES[k]} {pct(v, 0)}" for k, v in s.items())
            lineas.append(f"| {NOMBRES[g]} | {MODELOS[modelo]} | {texto} |")
    return "\n".join(lineas)


def tabla_por_genero(escenario: str = "C") -> str:
    df = sensibilidad_por_genero(escenario)
    lineas = ["| Género | " + " | ".join(df.columns) + " |", "| --- |" + " ---: |" * len(df.columns)]
    for g, fila in df.iterrows():
        lineas.append(f"| {g} | " + " | ".join(pct(v, 0) for v in fila) + " |")
    return "\n".join(lineas)


def tabla_busqueda_markdown() -> str:
    df = tabla_busqueda()
    lineas = ["| Modelo | Dropout | Tasa de aprendizaje | Mejor época | Validación por segmento (%) | Validación por canción (%) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in df.itertuples(index=False):
        dropout, tasa = (f"{v:g}".replace(".", ",") for v in (r[1], r[2]))
        lineas.append(f"| {r[0]} | {dropout} | {tasa} | {r[3]} | {pct(r[4])} | {pct(r[5])} |")
    return "\n".join(lineas)


def pct(x: float, decimales: int = 1) -> str:
    """Porcentaje con coma decimal: 0.8312 -> '83,1'."""
    return f"{100 * x:.{decimales}f}".replace(".", ",")


def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def tabla_markdown(df: pd.DataFrame | None = None) -> str:
    df = resumen() if df is None else df
    lineas = ["| Escenario | Géneros | Modelo | Parámetros | Exactitud por segmento (%) | Exactitud por canción (%) |",
              "| --- | ---: | --- | ---: | ---: | ---: |"]
    for r in df.itertuples():
        lineas.append(f"| {r.escenario} | {len(ESCENARIOS[r.escenario])} | {MODELOS[r.modelo]} | {miles(r.parametros)} "
                      f"| {pct(r.segmento_media)} ± {pct(r.segmento_de)} | {pct(r.cancion_media)} ± {pct(r.cancion_de)} |")
    return "\n".join(lineas)

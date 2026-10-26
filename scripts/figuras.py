"""Genera las figuras de results/figuras/ a partir de results/ y del audio local.

Uso:
    python scripts/figuras.py
"""

import pandas as pd

from generos import datos, figuras, proyeccion
from generos import resultados as res
from generos.config import ESCENARIOS, FIGURAS


def ejemplos_por_genero(escenario: str = "C") -> dict[str, str]:
    """Primera canción de prueba (en orden alfabético) de cada género."""
    cat = datos.catalogo()
    prueba = cat[cat["particion"] == "test"].sort_values("cancion")
    return {g: prueba[prueba["genero"] == g]["ruta"].iloc[0] for g in ESCENARIOS[escenario]}


def main() -> None:
    figuras.aplicar_estilo()
    figuras.espectrogramas(ejemplos_por_genero(), FIGURAS / "espectrogramas.png")

    resumen = res.resumen()
    figuras.exactitud(resumen, FIGURAS / "exactitud.png")
    for e in resumen["escenario"].unique():
        for m in res.MODELOS:
            cm, generos = res.confusion_media(e, m)
            figuras.confusion(cm, generos, f"{res.MODELOS[m]} · escenario {e}", FIGURAS / f"confusion_{e}_{m}.png")

    if not proyeccion.ruta_mapa("C", "lstm").exists():
        proyeccion.calcular("C", "lstm")
    mapa = proyeccion.cargar("C", "lstm")
    xy, generos = mapa[["x", "y"]].to_numpy(), mapa["genero"].to_numpy()
    figuras.mapa_generos(xy, generos, list(ESCENARIOS["C"]), FIGURAS / "mapa_generos.png")
    figuras.mapa_pareja(xy, generos, ("rock", "rock_es"), FIGURAS / "mapa_rock.png")
    print(pd.Series(sorted(p.name for p in FIGURAS.glob("*.png"))).to_string(index=False))


if __name__ == "__main__":
    main()

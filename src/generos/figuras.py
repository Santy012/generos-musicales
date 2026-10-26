"""Figuras con estilo común. Colores fijos por modelo; rampas de un solo tono para magnitudes.

La paleta de modelos (azul y naranja) se validó para daltonismo entre todos los pares.
Los mapas 2D de muchos géneros se dibujan como paneles pequeños con un género
resaltado sobre el resto en gris, en lugar de 14 colores indistinguibles.
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from generos.config import ESCENARIOS, NOMBRES, SAMPLE_RATE

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
GRIS_PUNTOS = "#d6d5ce"
COLORES = {"densa": "#2a78d6", "lstm": "#eb6834"}
DESTACADO = ("#2a78d6", "#eb6834")
RAMPA = LinearSegmentedColormap.from_list(
    "azul", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MODELOS = {"densa": "Red densa", "lstm": "LSTM"}


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": TINTA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "xtick.color": TENUE, "ytick.color": TENUE, "axes.edgecolor": EJE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": REJILLA, "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def guardar(fig, ruta) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def espectrogramas(ejemplos: dict[str, str], ruta=None, columnas: int = 7):
    """Espectrograma mel de 5 s (de 10 a 15 s) de una canción por género."""
    generos = list(ejemplos)
    filas = int(np.ceil(len(generos) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(2.1 * columnas, 1.9 * filas), sharex=True, sharey=True)
    for ax, g in zip(ejes.flat, generos):
        y, _ = librosa.load(ejemplos[g], sr=SAMPLE_RATE, offset=10, duration=5)
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64), ref=np.max)
        ax.imshow(S, origin="lower", aspect="auto", cmap=RAMPA, vmin=-80, vmax=0,
                  extent=[0, 5, 0, S.shape[0]])
        ax.set_title(NOMBRES[g], fontsize=9.5)
        ax.set_xticks([0, 5])
        ax.set_yticks([])
    for ax in ejes.flat[len(generos):]:
        ax.axis("off")
    fig.supxlabel("Tiempo (s)", fontsize=9, color=TINTA_2)
    fig.supylabel("Bandas mel (graves abajo)", fontsize=9, color=TINTA_2)
    fig.tight_layout()
    if ruta:
        guardar(fig, ruta)
    return fig


def confusion(cm: np.ndarray, generos: list[str], titulo: str, ruta=None, ax=None):
    """Matriz normalizada por fila: cada fila suma 100 % (sensibilidad en la diagonal)."""
    nuevo = ax is None
    if nuevo:
        lado = 0.42 * len(generos) + 1.6
        fig, ax = plt.subplots(figsize=(lado + 0.6, lado))
    nombres = [NOMBRES[g] for g in generos]
    im = ax.imshow(cm * 100, cmap=RAMPA, vmin=0, vmax=100)
    for i in range(len(generos)):
        for j in range(len(generos)):
            v = 100 * cm[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v >= 55 else TINTA)
    ax.set_xticks(range(len(nombres)), nombres, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(nombres)), nombres, fontsize=8.5)
    ax.set_xlabel("Género predicho")
    ax.set_ylabel("Género real")
    ax.set_title(titulo)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    if nuevo:
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="% de las canciones del género real")
        fig.tight_layout()
        if ruta:
            guardar(fig, ruta)
        return fig
    return im


def exactitud(resumen, ruta=None):
    """Exactitud en prueba por escenario y modelo: media ± desviación entre semillas."""
    fig, ejes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    escenarios = sorted(resumen["escenario"].unique())
    x = np.arange(len(escenarios))
    ancho = 0.36
    for ax, (nivel, titulo) in zip(ejes, [("segmento", "Por segmento de 5 s"), ("cancion", "Por canción (voto)")]):
        for k, modelo in enumerate(MODELOS):
            d = resumen[resumen["modelo"] == modelo].set_index("escenario").loc[escenarios]
            pos = x + (k - 0.5) * (ancho + 0.03)
            ax.bar(pos, 100 * d[f"{nivel}_media"], ancho, color=COLORES[modelo], label=MODELOS[modelo], zorder=2)
            ax.errorbar(pos, 100 * d[f"{nivel}_media"], yerr=100 * d[f"{nivel}_de"], fmt="none",
                        ecolor=TINTA_2, elinewidth=1, capsize=3, zorder=3)
            for p, v in zip(pos, 100 * d[f"{nivel}_media"]):
                ax.text(p, 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                        color="#ffffff", zorder=4)
        ax.set_xticks(x, [f"{e} · {len(ESCENARIOS[e])} géneros" for e in escenarios])
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, zorder=0)
        ax.set_title(titulo)
    ejes[0].set_ylabel("Exactitud en prueba (%)")
    ejes[1].legend(loc="upper right")
    fig.tight_layout()
    if ruta:
        guardar(fig, ruta)
    return fig


def mapa_generos(xy: np.ndarray, generos: np.ndarray, orden: list[str], ruta=None, columnas: int = 7):
    """Mapa 2D en paneles: cada panel resalta un género sobre el resto en gris."""
    filas = int(np.ceil(len(orden) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(1.95 * columnas, 2.0 * filas), sharex=True, sharey=True)
    for ax, g in zip(ejes.flat, orden):
        dentro = generos == g
        ax.scatter(*xy[~dentro].T, s=2, c=GRIS_PUNTOS, linewidths=0, rasterized=True)
        ax.scatter(*xy[dentro].T, s=4, c=DESTACADO[0], linewidths=0, rasterized=True)
        ax.set_title(NOMBRES[g], fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    for ax in ejes.flat[len(orden):]:
        ax.axis("off")
    fig.tight_layout()
    if ruta:
        guardar(fig, ruta)
    return fig


def mapa_pareja(xy: np.ndarray, generos: np.ndarray, pareja: tuple[str, str], ruta=None):
    """Mapa 2D con dos géneros resaltados (p. ej., rock y rock en español)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    otros = ~np.isin(generos, pareja)
    ax.scatter(*xy[otros].T, s=3, c=GRIS_PUNTOS, linewidths=0, label="Otros géneros", rasterized=True)
    for g, color in zip(pareja, DESTACADO):
        d = generos == g
        ax.scatter(*xy[d].T, s=9, c=color, edgecolors=SUPERFICIE, linewidths=0.4, label=NOMBRES[g])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.legend(loc="best", markerscale=1.6)
    fig.tight_layout()
    if ruta:
        guardar(fig, ruta)
    return fig

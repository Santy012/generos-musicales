"""Mapa 2D de la representación aprendida (t-SNE sobre la penúltima capa)."""

import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from generos import RANDOM_STATE, datos, modelos
from generos.config import DATA, RESULTS


def ruta_mapa(escenario: str, modelo: str):
    return RESULTS / f"mapa_{escenario}_{modelo}.csv"


def calcular(escenario: str, modelo: str) -> pd.DataFrame:
    """Proyecta los segmentos de prueba con el modelo guardado de la semilla 0."""
    archivo = next((DATA / "modelos").glob(f"{escenario}_{modelo}_*_s0.keras"))
    red = tf.keras.models.load_model(archivo)
    c = datos.conjunto(escenario)
    z = modelos.representacion(red).predict(c.test.X, verbose=0)
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE).fit_transform(z)
    df = pd.DataFrame({"x": xy[:, 0].round(3), "y": xy[:, 1].round(3),
                       "genero": c.etiquetas.nombres(c.test.y), "cancion": c.test.cancion})
    df.to_csv(ruta_mapa(escenario, modelo), index=False, lineterminator="\n")
    return df


def cargar(escenario: str, modelo: str) -> pd.DataFrame:
    return pd.read_csv(ruta_mapa(escenario, modelo))


def vecinos_mismo_genero(df: pd.DataFrame, k: int = 10) -> pd.Series:
    """Fracción de los k vecinos más cercanos en el mapa que son del mismo género."""
    xy = df[["x", "y"]].to_numpy()
    idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    g = df["genero"].to_numpy()
    return pd.Series((g[idx] == g[:, None]).mean(1), index=df.index).groupby(df["genero"]).mean()

"""Métricas por segmento y por canción (voto de sus segmentos)."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def voto(prob: np.ndarray, cancion: np.ndarray) -> pd.DataFrame:
    """Predicción por canción: el género más votado por sus segmentos.

    Los empates se resuelven con la probabilidad media de los segmentos.
    """
    df = pd.DataFrame({"cancion": cancion, "pred": prob.argmax(1)})
    filas = []
    for c, idx in df.groupby("cancion", sort=False).indices.items():
        votos = np.bincount(prob[idx].argmax(1), minlength=prob.shape[1]).astype(float)
        votos += prob[idx].mean(0) * 1e-3  # desempate
        filas.append({"cancion": c, "pred": int(votos.argmax())})
    return pd.DataFrame(filas)


def metricas(prob: np.ndarray, y: np.ndarray, cancion: np.ndarray, n_clases: int) -> dict:
    pred_seg = prob.argmax(1)
    por_cancion = voto(prob, cancion)
    real = pd.Series(y, index=cancion).groupby(level=0).first()
    por_cancion["real"] = real.loc[por_cancion["cancion"]].to_numpy()
    clases = list(range(n_clases))
    cm = confusion_matrix(por_cancion["real"], por_cancion["pred"], labels=clases)
    return {
        "exactitud_segmento": float((pred_seg == y).mean()),
        "exactitud_cancion": float((por_cancion["pred"] == por_cancion["real"]).mean()),
        "n_segmentos": int(len(y)),
        "n_canciones": int(len(por_cancion)),
        "confusion_cancion": cm.tolist(),
        "confusion_segmento": confusion_matrix(y, pred_seg, labels=clases).tolist(),
        "predicciones_cancion": por_cancion.to_dict(orient="list"),
    }

"""Rutas, géneros y escenarios: la única fuente de verdad del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
DATA_SOURCES = RAIZ / "data_sources"
RESULTS = RAIZ / "results"
FIGURAS = RESULTS / "figuras"
LOGS = RAIZ / "logs"

# Formato común de audio (el de GTZAN) y segmentación.
SAMPLE_RATE = 22_050
DURACION_CANCION = 30  # segundos
DURACION_SEGMENTO = 5  # segundos
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

GENEROS_GTZAN = ("blues", "classical", "country", "disco", "hiphop",
                 "jazz", "metal", "pop", "reggae", "rock")
GENEROS_NUEVOS = ("salsa", "vallenato", "reggaeton", "rock_es")

# Orden de las clases de cada escenario. El índice de cada género sale de aquí.
ESCENARIOS = {
    "A": GENEROS_GTZAN,
    "B": GENEROS_GTZAN + ("salsa", "vallenato"),
    "C": GENEROS_GTZAN + GENEROS_NUEVOS,
}

# Nombres para tablas y figuras.
NOMBRES = {
    "blues": "Blues", "classical": "Clásica", "country": "Country", "disco": "Disco",
    "hiphop": "Hip hop", "jazz": "Jazz", "metal": "Metal", "pop": "Pop",
    "reggae": "Reggae", "rock": "Rock", "salsa": "Salsa", "vallenato": "Vallenato",
    "reggaeton": "Reguetón", "rock_es": "Rock en español",
}

"""Partición fault-filtered de GTZAN: tamaños, exclusiones y ninguna canción compartida."""

import pytest

from generos import gtzan
from generos.config import GENEROS_GTZAN


@pytest.fixture(scope="module")
def df():
    return gtzan.particion()


def test_tamanos(df):
    assert df["particion"].value_counts().to_dict() == {"train": 442, "test": 290, "valid": 197}


def test_excluye_archivo_danado(df):
    assert "jazz.00054" not in set(df["cancion"])


def test_ninguna_cancion_en_dos_particiones(df):
    assert df["cancion"].is_unique
    assert df["grupo"].is_unique


def test_todos_los_generos_en_cada_particion(df):
    for _, g in df.groupby("particion"):
        assert set(g["genero"]) == set(GENEROS_GTZAN)


def test_genero_coincide_con_el_archivo(df):
    assert (df["cancion"].str.split(".").str[0] == df["genero"]).all()

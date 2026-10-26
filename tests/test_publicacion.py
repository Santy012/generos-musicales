"""El README coincide con results/ y los notebooks corren de principio a fin."""

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((RAIZ / "notebooks").glob("*.ipynb"))


def test_readme_al_dia():
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "actualizar_readme.py"), "--comprobar"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.notebooks
@pytest.mark.parametrize("ruta", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_corre(ruta):
    nb = nbformat.read(ruta, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(ruta.parent)}}).execute()

"""Petit utilitaire de construction de notebooks Jupyter.

Les notebooks du projet sont generes a partir de scripts Python afin de garantir
leur reproductibilite : le contenu est versionnable en texte, et l'execution est
faite ensuite par `jupyter nbconvert --execute`, ce qui garantit que toutes les
sorties affichees proviennent d'une execution reelle.
"""
import json
import uuid
from pathlib import Path


def _uid():
    return uuid.uuid4().hex[:12]


def md(text):
    """Cellule Markdown."""
    return {"cell_type": "markdown", "id": _uid(), "metadata": {}, "source": text.rstrip("\n").split("\n")}


def code(text):
    """Cellule de code."""
    return {"cell_type": "code", "id": _uid(), "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").split("\n")}


def write(path, cells, kernel="python3"):
    nb = {
        "cells": [{**c, "source": [l + "\n" for l in c["source"][:-1]] + [c["source"][-1]]}
                  for c in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
            "language_info": {"name": "python", "version": "3.10.12",
                              "mimetype": "text/x-python",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                              "file_extension": ".py"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"ecrit : {path}  ({len(cells)} cellules)")

"""Assemble le notebook unique du projet a partir des modules de section.

Usage :
    python tools/build_notebook.py            # assemble tout le pipeline
    python tools/build_notebook.py 0 1 2      # assemble seulement ces sections

Le notebook produit est ensuite execute par :
    jupyter nbconvert --to notebook --execute --inplace notebooks/house_prices_pipeline.ipynb
"""
import importlib
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "tools"))
sys.path.insert(0, str(RACINE / "tools" / "sections"))

from nbgen import write  # noqa: E402

SECTIONS = {
    0: "s0_intro",
    1: "s1_data",
    2: "s2_eda",
    3: "s3_preprocessing",
    4: "s4_transformation",
    5: "s5_machine_learning",
    6: "s6_evaluation",
    7: "s7_deployment",
}

SORTIE = RACINE / "notebooks" / "house_prices_pipeline.ipynb"


def main(demandees=None):
    numeros = sorted(demandees) if demandees else sorted(SECTIONS)
    cellules = []
    for n in numeros:
        nom = SECTIONS[n]
        chemin = RACINE / "tools" / "sections" / f"{nom}.py"
        if not chemin.exists():
            print(f"  section {n} ({nom}) : pas encore ecrite, ignoree")
            continue
        module = importlib.import_module(f"sections.{nom}")
        importlib.reload(module)
        bloc = module.cells()
        cellules.extend(bloc)
        print(f"  section {n} ({nom}) : {len(bloc)} cellules")

    write(SORTIE, cellules)
    return SORTIE


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or None
    main(args)

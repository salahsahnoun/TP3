"""Assemble le notebook unique du projet 2 a partir des modules de section."""
import importlib
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "tools"))
sys.path.insert(0, str(RACINE / "tools" / "sections"))

from nbgen import write  # noqa: E402

SECTIONS = {
    1: "s1_bigdata",
    2: "s2_hadoop_spark",
    3: "s3_transformation",
    4: "s4_machine_learning",
    5: "s5_evaluation",
    6: "s6_deployment",
}
SORTIE = RACINE / "notebooks" / "nyc_taxi_pipeline.ipynb"


def main(demandees=None):
    cellules = []
    for n in sorted(demandees or SECTIONS):
        nom = SECTIONS[n]
        if not (RACINE / "tools" / "sections" / f"{nom}.py").exists():
            print(f"  section {n} ({nom}) : pas encore ecrite")
            continue
        module = importlib.import_module(f"sections.{nom}")
        importlib.reload(module)
        bloc = module.cells()
        cellules.extend(bloc)
        print(f"  section {n} ({nom}) : {len(bloc)} cellules")
    write(SORTIE, cellules)
    return SORTIE


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or None)

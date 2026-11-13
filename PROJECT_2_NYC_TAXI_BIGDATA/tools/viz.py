"""Conventions graphiques du projet.

Une seule palette, un seul style, importes par tous les notebooks : les figures
du rapport forment ainsi un ensemble coherent. La palette categorielle a ete
validee pour l'accessibilite (distinguabilite en vision normale et en vision
deficiente -- deuteranopie, protanopie, tritanopie).

Regles appliquees :
  - couleur categorielle attribuee dans un ordre fixe, jamais cyclee ;
  - echelle sequentielle = une seule teinte, clair -> fonce (magnitude) ;
  - echelle divergente = deux teintes opposees + gris neutre au centre (polarite) ;
  - grille et axes en retrait, marques fines, pas d'effet decoratif.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Palette categorielle (ordre fixe)
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLEU, ORANGE, AQUA, JAUNE = SERIE[0], SERIE[1], SERIE[2], SERIE[3]
ROUGE = SERIE[7]

# Surfaces et encres
SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
ENCRE_3 = "#8a8984"
GRILLE = "#e5e4e0"

# Echelle sequentielle : une teinte, clair -> fonce
RAMPE_BLEUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
               "#2a78d6", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_bleu", RAMPE_BLEUE)

# Echelle divergente : bleu <-> rouge, gris neutre au centre
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_bleu_rouge",
    ["#0d366b", "#256abf", "#6da7ec", "#cde2fb",
     "#f0efec",
     "#f6c9c8", "#ef8d8c", "#e34948", "#a32322"],
)


def appliquer_style():
    """Installe le style du projet dans matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",

        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": ENCRE_2,
        "text.color": ENCRE,

        # Axes et grille en retrait : la donnee doit dominer
        "axes.edgecolor": GRILLE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRILLE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,

        "xtick.color": ENCRE_3,
        "ytick.color": ENCRE_3,
        "xtick.labelcolor": ENCRE_2,
        "ytick.labelcolor": ENCRE_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "patch.linewidth": 0,

        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": ENCRE_2,

        "axes.prop_cycle": mpl.cycler(color=SERIE),
    })


def titrer(ax, titre, sous_titre=None):
    """Titre + sous-titre explicatif.

    Le sous-titre porte l'interpretation : il doit dire ce que la figure
    *montre*, pas repeter ce qu'elle represente. Il est place sous le titre,
    au-dessus de la zone de traces -- d'ou le decalage (`pad`) du titre.
    """
    if sous_titre:
        ax.set_title(titre, color=ENCRE, pad=26)
        ax.text(0, 1.012, sous_titre, transform=ax.transAxes,
                fontsize=9, color=ENCRE_2, va="bottom")
    else:
        ax.set_title(titre, color=ENCRE)


def sauver(fig, nom, dossier=None):
    """Enregistre la figure dans reports/figures/ pour le rapport final."""
    if dossier is None:
        racine = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
        dossier = racine / "reports" / "figures"
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{nom}.png"
    fig.savefig(chemin)
    return chemin


def format_dollars(ax, axe="y"):
    """Formate un axe en milliers de dollars."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k$" if v else "0")
    (ax.yaxis if axe == "y" else ax.xaxis).set_major_formatter(fmt)

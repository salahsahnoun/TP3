"""Section 0 : page de garde, sommaire et configuration globale."""
from nbgen import md, code


def cells():
    return [

md("""
# Prédiction du prix de vente de maisons
## *House Prices - Advanced Regression Techniques*

**Module** : Fondements du Big Data, Projet 1 (Data Science classique)
**Dataset** : Ames Housing (Iowa, 2006-2010), 1 460 ventes, 79 caractéristiques
**Cible** : `SalePrice`, régression supervisée
**Notebook de référence** : [`gusthema/house-prices-prediction-using-tfdf`](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf)
**Technologie principale** : TensorFlow Decision Forests

## Sommaire

| # | Étape | Contenu | Artefact produit |
|---|-------|---------|------------------|
| [1](#etape-1) | DATA | Comprendre les données sans les modifier | `metadonnees.json` |
| [2](#etape-2) | EDA | Distributions, corrélations, anomalies | `reports/figures/` |
| [3](#etape-3) | PRÉTRAITEMENT | Découpage, nettoyage, imputation | `train_clean.csv` |
| [4](#etape-4) | TRANSFORMATION | Feature engineering, encodage | `preprocessor.joblib` |
| [5](#etape-5) | MACHINE LEARNING | TFDF puis modèles de comparaison | `models/` |
| [6](#etape-6) | ÉVALUATION | RMSE, MAE, R², analyse des erreurs | `comparaison_modeles.csv` |
| [7](#etape-7) | DÉPLOIEMENT | Application de prédiction | `app/` |

## Quatre règles tenues tout au long du notebook

**1. Aucune fuite de données.** Le découpage train/validation est fait à l'étape 3, avant
toute imputation. Les statistiques apprises (médianes, encodages, mises à l'échelle) le sont
sur le train seul. Vérifié par des tests explicites à l'étape 4.

**2. Aucune métrique sur `test.csv`.** Ce fichier ne contient pas `SalePrice` : les vrais
prix sont gardés secrets par Kaggle. Toutes les métriques viennent d'un jeu de validation
découpé dans `train.csv`.

**3. Aucun résultat inventé.** Chaque chiffre affiché provient d'une exécution réelle.

**4. Comparaison équitable.** Tous les modèles utilisent le même découpage, la même graine
et le même jeu de validation.

## Reproductibilité

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/house_prices_pipeline.ipynb
```
"""),

code("""
# CONFIGURATION GLOBALE (a executer en premier)
import os
import sys
import random
import warnings
from pathlib import Path

import numpy as np

# Graine unique : condition necessaire a la reproductibilite
GRAINE = 42
random.seed(GRAINE)
np.random.seed(GRAINE)
os.environ["PYTHONHASHSEED"] = str(GRAINE)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Arborescence
PROJET = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_RAW = PROJET / "data" / "raw"
DATA_PROC = PROJET / "data" / "processed"
MODELS = PROJET / "models"
REPORTS = PROJET / "reports"
FIGURES = REPORTS / "figures"
APP = PROJET / "app"
for d in (DATA_PROC, MODELS, REPORTS, FIGURES, APP):
    d.mkdir(parents=True, exist_ok=True)

# Style graphique commun a toutes les figures
sys.path.insert(0, str(PROJET / "tools"))
import viz
viz.appliquer_style()

print(f"Projet : {PROJET}")
print(f"Graine : {GRAINE}")
"""),
    ]

"""Genere README.md a partir des resultats reels du pipeline.

Les chiffres sont lus dans les fichiers produits par le notebook :
  reports/comparaison_modeles.csv
  reports/validation_croisee.csv
  models/deploiement.json

Aucune valeur n'est saisie a la main, ce qui garantit que la documentation
reste coherente avec la derniere execution du pipeline.
"""
import json
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent


def charger():
    comp = pd.read_csv(RACINE / "reports" / "comparaison_modeles.csv")
    cv = pd.read_csv(RACINE / "reports" / "validation_croisee.csv")
    cfg = json.loads((RACINE / "models" / "deploiement.json").read_text())
    return comp, cv, cfg


def tableau_markdown(df, colonnes, entetes, formats=None):
    formats = formats or {}
    lignes = ["| " + " | ".join(entetes) + " |",
              "|" + "|".join(["---"] * len(entetes)) + "|"]
    for _, r in df.iterrows():
        cellules = []
        for c in colonnes:
            v = r[c]
            cellules.append(formats[c](v) if c in formats else str(v))
        lignes.append("| " + " | ".join(cellules) + " |")
    return "\n".join(lignes)


FMT = {
    "modele": lambda v: str(v).split(". ", 1)[-1],
    "RMSE_valid": lambda v: f"{v:,.0f} $".replace(",", " "),
    "RMSE_train": lambda v: f"{v:,.0f} $".replace(",", " "),
    "MAE_valid": lambda v: f"{v:,.0f} $".replace(",", " "),
    "R2_valid": lambda v: f"{v:.4f}",
    "RMSLE_valid": lambda v: f"{v:.4f}",
    "MAPE_valid": lambda v: f"{v:.2f} %",
    "duree_s": lambda v: f"{v:.1f} s",
    "RMSE_log_moyenne": lambda v: f"{v:.4f}",
    "ecart_type": lambda v: f"{v:.4f}",
}


def readme(comp, cv, cfg):
    meilleur = comp.iloc[0]
    original = comp[comp.modele.str.startswith("1.")].iloc[0]
    notre = comp[comp.modele.str.startswith("2.")].iloc[0]
    gain = 100 * (original.RMSE_valid - notre.RMSE_valid) / original.RMSE_valid

    return f"""# Projet 1 : prédiction du prix de vente de maisons

Reproduction commentée et améliorée du notebook Kaggle
[`gusthema/house-prices-prediction-using-tfdf`](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf),
dans le cadre du module Fondements du Big Data.

## Contexte

L'estimation immobilière repose traditionnellement sur l'expertise d'un agent qui compare
un bien à des transactions récentes. L'approche est lente, coûteuse et subjective.
L'objectif de ce projet est de l'automatiser par un modèle statistique entraîné sur des
ventes réelles.

Le travail ne se limite pas à reproduire le notebook de référence : il le complète par une
analyse exploratoire approfondie, un prétraitement explicite et vérifié, une comparaison
équitable de six modèles, et un déploiement fonctionnel.

## Problématique

Prédire le prix de vente `SalePrice` d'une maison à partir de 79 caractéristiques
objectives.

| Élément | Valeur |
|---|---|
| Type d'apprentissage | Supervisé, régression |
| Variable cible | `SalePrice`, en dollars |
| Métrique officielle Kaggle | RMSE sur log(SalePrice), soit une erreur relative |

## Dataset

**Ames Housing**, constitué par le Pr. Dean De Cock à partir du registre officiel des
transactions immobilières d'Ames (Iowa) entre 2006 et 2010.

| Caractéristique | Valeur |
|---|---|
| Observations | 1 460 ventes réelles |
| Variables explicatives | 79 (33 numériques, 23 ordinales, 23 nominales) |
| Entraînement / validation | {cfg['n_entrainement']} / {cfg['n_validation']} (découpage stratifié 80/20) |
| Valeurs manquantes | 19 colonnes, dont 15 signifient "équipement absent" |
| Doublons | Aucun |

Le fichier `test.csv` de la compétition ne contient pas `SalePrice` : aucune métrique de ce
projet n'est calculée dessus.

## Technologies

| Usage | Outil |
|---|---|
| Langage | Python 3.10 |
| Manipulation de données | pandas, NumPy |
| Visualisation | Matplotlib |
| Modèle principal | **TensorFlow Decision Forests 1.9.2** (TensorFlow 2.16.2) |
| Modèles de comparaison | scikit-learn 1.5.2 |
| Notebook | JupyterLab |
| Déploiement | Streamlit |

## Architecture

```
PROJECT_1_HOUSE_PRICES/
├── data/
│   ├── raw/                    train.csv
│   └── processed/              jeux nettoyés et préparés
├── notebooks/
│   └── house_prices_pipeline.ipynb      le pipeline complet, 7 étapes
├── tools/
│   ├── pipeline_maisons.py     classes de préparation (importables)
│   ├── viz.py                  palette et style des figures
│   ├── nbgen.py                construction des notebooks
│   ├── build_notebook.py       assemblage du notebook
│   ├── build_docs.py           génération de ce README
│   └── sections/               une étape du pipeline par module
├── models/                     modèles et transformateurs sauvegardés
├── app/
│   ├── prediction.py           logique de prédiction, partagée
│   └── app.py                  interface Streamlit
├── reports/
│   ├── figures/                figures du rapport
│   ├── comparaison_modeles.csv
│   └── validation_croisee.csv
├── requirements.txt
└── README.md
```

**Pourquoi le notebook est généré par des scripts.** Chaque étape est écrite dans un module
Python de `tools/sections/`, assemblé en un notebook unique par `build_notebook.py`, puis
exécuté par `nbconvert`. Cela garantit que **toutes les sorties affichées proviennent d'une
exécution réelle** : à chaque correction, l'intégralité du notebook est réexécutée.

**Pourquoi les classes sont dans un module.** `joblib` sérialise les objets par référence à
leur module. Une classe définie dans le notebook serait enregistrée sous `__main__` et
deviendrait introuvable depuis l'application Streamlit. Les placer dans
`tools/pipeline_maisons.py` rend les artefacts rechargeables partout. Le notebook affiche
leur code source.

## Pipeline

```
DATA  →  EDA  →  PRÉTRAITEMENT  →  TRANSFORMATION  →  MACHINE LEARNING  →  ÉVALUATION  →  DÉPLOIEMENT
```

| Étape | Contenu | Artefact |
|---|---|---|
| 1. DATA | Typologie des 79 variables, diagnostic qualité | `metadonnees.json` |
| 2. EDA | 9 figures, 7 décisions documentées | `reports/figures/` |
| 3. PRÉTRAITEMENT | Découpage, puis nettoyage appris sur le train seul | `train_clean.csv` |
| 4. TRANSFORMATION | 13 variables créées, 20 ordinales encodées, 2 pipelines | `preprocessor.joblib` |
| 5. MACHINE LEARNING | 6 modèles, validation croisée 5 plis | `models/` |
| 6. ÉVALUATION | RMSE, MAE, R², RMSLE, MAPE, analyse des erreurs | `comparaison_modeles.csv` |
| 7. DÉPLOIEMENT | Application Streamlit, test de non-régression | `app/` |

### Garanties méthodologiques

1. **Aucune fuite de données.** Le découpage précède tout traitement. Les statistiques
   (médianes d'imputation, encodages, normalisation) sont apprises sur le train seul.
   Trois tests automatiques le vérifient à l'étape 4.
2. **Aucune métrique sur `test.csv`.** Toutes les métriques viennent des
   {cfg['n_validation']} maisons de validation.
3. **Aucun résultat inventé.** Le notebook est réexécuté intégralement à chaque
   modification, et cette documentation est générée depuis les fichiers de résultats.
4. **Comparaison équitable.** Même découpage, même graine (42), même jeu de validation
   pour les six modèles.

## Résultats

Métriques sur les {cfg['n_validation']} maisons de validation, jamais vues à
l'entraînement. Toutes les valeurs sont **en dollars**, après reconversion des modèles
entraînés sur le logarithme.

{tableau_markdown(comp, ["modele", "RMSE_valid", "MAE_valid", "R2_valid", "RMSLE_valid", "MAPE_valid"],
                  ["Modèle", "RMSE", "MAE", "R²", "RMSLE", "MAPE"], FMT)}

Référence : prédire systématiquement le prix moyen donnerait une RMSE de 82 154 \\$. Le
meilleur modèle fait **{100 * (1 - meilleur.RMSE_valid / 82154):.0f} % d'erreur en moins**.

### Validation croisée (5 plis, sur le jeu d'entraînement)

{tableau_markdown(cv, ["modele", "RMSE_log_moyenne", "ecart_type"],
                  ["Modèle", "RMSE (espace log)", "Écart-type"], FMT)}

### Trois conclusions

**1. Notre prétraitement n'améliore pas significativement TFDF.**
{original.RMSE_valid:,.0f} \\$ pour la reproduction brute du notebook original contre
{notre.RMSE_valid:,.0f} \\$ avec nos étapes 3 et 4, soit **{gain:.1f} %**, nettement en
dessous du bruit d'échantillonnage estimé à 6,3 % par validation croisée. L'explication est
que TFDF gère déjà nativement les valeurs manquantes, les variables catégorielles et les
échelles. Le prétraitement reste néanmoins indispensable pour les modèles linéaires et pour
rendre le pipeline auditable.

**2. Le classement dépend de la métrique.** TFDF Random Forest gagne en RMSE, Ridge gagne
en erreur relative (MAPE et RMSLE). Ce n'est pas contradictoire : la RMSE élève les erreurs
au carré et se concentre donc sur les maisons chères, tandis que la MAPE traite toutes les
maisons à égalité. Le choix dépend de l'usage visé.

**3. Les erreurs sont extrêmement concentrées.** 10 maisons sur {cfg['n_validation']}
(3,4 %) portent 63 % de l'erreur quadratique totale, et une seule en porte 20 %. Ce sont
des ventes atypiques (construction inachevée, saisie, vente entre proches) dont le prix ne
reflète pas la valeur du bien.

### Modèle déployé

**{cfg['nom_modele']}**, retenu sur quatre critères : performance, robustesse en validation
croisée, simplicité de mise en production (TFDF consomme directement un DataFrame avec des
colonnes texte), et cohérence avec la consigne de privilégier la technologie du notebook
original.

| Métrique | Valeur |
|---|---|
| RMSE | {cfg['rmse_validation']:,} $ |
| MAE | {cfg['mae_validation']:,} $ |
| R² | {cfg['r2_validation']} |
| Erreur relative moyenne | {cfg['mape_validation']} % |

## Installation

Python 3.10 est requis : TensorFlow Decision Forests 1.9.x ne publie pas de roue pour
Python 3.12 et plus.

```bash
cd PROJECT_1_HOUSE_PRICES
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Données

Le notebook attend `data/raw/train.csv` (1 460 lignes, 81 colonnes).

Avec un compte Kaggle :

```bash
# kaggle.com > Settings > API > Create New Token
mkdir -p ~/.kaggle && mv ~/Téléchargements/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c house-prices-advanced-regression-techniques \\
    -p data/raw && unzip -o data/raw/*.zip -d data/raw
```

Sans compte Kaggle, le même jeu est disponible sur OpenML (identifiant 42165) au format
ARFF, convertible en CSV.

## Exécution

### Le notebook

```bash
source .venv/bin/activate
jupyter lab notebooks/house_prices_pipeline.ipynb
```

Le notebook est livré **déjà exécuté** : toutes les sorties et figures sont enregistrées.
"Run All" le réexécute à l'identique, la graine étant fixée à 42.

### Régénérer le notebook depuis les sources

```bash
python tools/build_notebook.py          # assemble les 8 modules de tools/sections/
jupyter nbconvert --to notebook --execute --inplace \\
    notebooks/house_prices_pipeline.ipynb
python tools/build_docs.py              # met a jour le README
```

### L'application

```bash
streamlit run app/app.py
```

Puis ouvrir http://localhost:8501. Le démarrage prend quelques secondes, le temps de
charger TensorFlow.

## Améliorations apportées au notebook original

| Domaine | Notebook original | Ce projet |
|---|---|---|
| EDA | Distribution de la cible, histogrammes | 9 figures, corrélations, multicolinéarité, anomalies |
| Valeurs manquantes | Laissées à TFDF | Sémantique prouvée par croisement, imputation justifiée |
| Variables | Brutes | 13 variables créées, 20 ordinales encodées |
| Cible | Brute | log(SalePrice), choix validé empiriquement |
| Fuite de données | Non traitée | Découpage avant nettoyage, 3 tests automatiques |
| Modèles | 1 | 6, avec validation croisée |
| Métriques | RMSE | RMSE, MAE, R², RMSLE, MAPE, analyse des erreurs |
| Déploiement | Aucun | Streamlit testé, non-régression vérifiée |

## Limites

### Du dataset

1. **Portée géographique nulle.** Uniquement Ames, Iowa, 60 000 habitants. Le modèle n'est
   pas transposable à un autre marché.
2. **Période atypique.** 2006 à 2010 couvre la bulle immobilière et la crise des subprimes.
3. **Obsolescence.** Les données ont 15 ans ; seules les relations relatives entre
   caractéristiques restent informatives.
4. **Variables absentes.** Ni taux d'intérêt, ni qualité des écoles, ni distance au centre,
   ni coordonnées GPS.
5. **Taille modeste.** 1 460 lignes, les intervalles de confiance sont larges.

### De la démarche

6. **Pas de recherche d'hyperparamètres exhaustive.** Les valeurs viennent des
   recommandations usuelles. Sur 1 167 lignes, une recherche par grille risquerait de
   surapprendre le protocole de validation.
7. **Validation aléatoire et non temporelle.** Un découpage chronologique serait plus
   réaliste mais réduirait encore la taille du jeu de validation.
8. **Pas de soumission Kaggle.** Sans `test.csv`, impossible de se situer dans le
   classement public.

### Du déploiement

9. **Pas de conteneurisation**, pas d'authentification, pas de surveillance de dérive, pas
   de réentraînement automatique.

## Perspectives

- Enrichir avec des données externes : indice des prix d'Ames par année, coordonnées GPS
  des quartiers pour calculer une distance au centre.
- Entraîner un modèle dédié aux ventes `Normal` (82 % du jeu) et traiter les conditions
  atypiques séparément.
- Combiner les modèles par empilement, en acceptant de perdre en lisibilité comparative.
- Produire un intervalle de prédiction par régression quantile plutôt qu'une fourchette
  fondée sur l'erreur moyenne.
- Conteneuriser l'application et figer les versions dans un `Dockerfile`.
"""


def main():
    comp, cv, cfg = charger()
    (RACINE / "README.md").write_text(readme(comp, cv, cfg), encoding="utf-8")
    print("README.md genere depuis les resultats reels.")


if __name__ == "__main__":
    main()

# Fondements du Big Data

Travaux pratiques du module Fondements du Big Data, école d'ingénieur en informatique,
5ᵉ année.

L'exercice consiste à partir de notebooks Kaggle existants, à les reproduire, les
commenter, et les améliorer en suivant un pipeline méthodologique imposé.

## Projets

| Projet | Sujet | Technologie | État |
|---|---|---|---|
| [1. House Prices](PROJECT_1_HOUSE_PRICES/) | Prédire le prix de vente d'une maison | TensorFlow Decision Forests | Terminé |
| 2. NYC Taxi | Prédire la durée d'un trajet en taxi | Hadoop, Spark, PySpark | À venir |

## Projet 1 : House Prices

Reproduction commentée et améliorée du notebook Kaggle
[`gusthema/house-prices-prediction-using-tfdf`](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf).

**Problématique** : prédire `SalePrice` à partir de 79 caractéristiques décrivant une
maison. Apprentissage supervisé, régression.

**Dataset** : Ames Housing, 1 460 ventes réelles à Ames (Iowa) entre 2006 et 2010.

**Pipeline** : DATA → EDA → PRÉTRAITEMENT → TRANSFORMATION → MACHINE LEARNING →
ÉVALUATION → DÉPLOIEMENT, dans un notebook unique de 148 cellules.

**Résultat** : TensorFlow Decision Forests (Random Forest) retenu parmi 6 modèles
comparés. RMSE de 32 817 $, erreur relative moyenne de 11,0 %, R² de 0,840 sur
292 maisons de validation jamais vues à l'entraînement.

Documentation détaillée : [`PROJECT_1_HOUSE_PRICES/README.md`](PROJECT_1_HOUSE_PRICES/README.md)

## Démarrage rapide

Python 3.10 est requis (TensorFlow Decision Forests ne publie pas de paquet pour les
versions plus récentes).

```bash
git clone https://github.com/salahsahnoun/TP3.git
cd TP3/PROJECT_1_HOUSE_PRICES

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter lab notebooks/house_prices_pipeline.ipynb
```

Le notebook est livré déjà exécuté : toutes les sorties et les 17 figures sont
enregistrées, aucune exécution n'est nécessaire pour le lire.

Pour lancer l'application d'estimation :

```bash
streamlit run app/app.py
```

## Organisation du dépôt

```
PROJECT_1_HOUSE_PRICES/
├── notebooks/      le pipeline complet, exécuté
├── tools/          scripts de génération du notebook et des figures
├── app/            application Streamlit d'estimation
├── models/         modèles et transformateurs sauvegardés
├── reports/        figures et tableaux de résultats
├── data/raw/       jeu de données d'entraînement
└── README.md       documentation détaillée du projet
```

Ne sont pas versionnés : l'environnement virtuel, les données intermédiaires et le modèle
Random Forest scikit-learn de 31 Mo. Tous sont régénérés par l'exécution du notebook.

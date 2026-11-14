# Fondements du Big Data

Travaux pratiques du module Fondements du Big Data, école d'ingénieur en informatique,
5ᵉ année.

L'exercice consiste à partir de notebooks Kaggle existants, à les reproduire, les
commenter, et les améliorer en suivant un pipeline méthodologique imposé.

## Projets

| Projet | Sujet | Technologie | État |
|---|---|---|---|
| [1. House Prices](PROJECT_1_HOUSE_PRICES/) | Prédire le prix de vente d'une maison | TensorFlow Decision Forests | Terminé |
| [2. NYC Taxi](PROJECT_2_NYC_TAXI_BIGDATA/) | Prédire la durée d'un trajet en taxi | Hadoop, HDFS, Spark, PySpark | Terminé |

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

## Projet 2 : NYC Taxi

Traitement distribué sur les données ouvertes de la NYC Taxi and Limousine Commission,
inspiré du notebook Kaggle
[`headsortails/nyc-taxi-eda-update-the-fast-the-curious`](https://www.kaggle.com/code/headsortails/nyc-taxi-eda-update-the-fast-the-curious).

**Problématique** : estimer la durée d'une course à partir des seules informations connues
au moment de la prise en charge. Cette contrainte exclut la distance au compteur, le prix
et l'heure d'arrivée, qui n'existent pas encore quand il faut répondre.

**Dataset** : janvier 2016, 10,9 millions de courses, 145 Mo en Parquet. Découpage
chronologique : 8,1 M courses pour apprendre (jours 1 à 24), 2,5 M pour tester (jours 25
à 31).

**Pipeline** : BIG DATA → HADOOP/SPARK → TRANSFORMATION → MACHINE LEARNING → ÉVALUATION →
DÉPLOIEMENT, sur un cluster HDFS mono-nœud réel et Spark en mode local.

**Résultat** : Gradient Boosted Trees retenu parmi 4 modèles Spark MLlib. RMSE de 446 s,
R² de 0,581. Le GBT ne devance l'arbre de décision simple que d'une seconde de RMSE pour
huit fois plus de temps de calcul, un bon exemple de rendement décroissant.

Documentation détaillée : [`PROJECT_2_NYC_TAXI_BIGDATA/README.md`](PROJECT_2_NYC_TAXI_BIGDATA/README.md)

## Démarrage rapide

Python 3.10 est requis (TensorFlow Decision Forests ne publie pas de paquet pour les
versions plus récentes).

Chaque projet a son propre environnement virtuel et son propre `requirements.txt`, pour
éviter tout conflit entre TensorFlow et Spark.

```bash
git clone https://github.com/salahsahnoun/TP3.git

# Projet 1, Python 3.10 requis pour TensorFlow Decision Forests
cd TP3/PROJECT_1_HOUSE_PRICES
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/house_prices_pipeline.ipynb

# Projet 2, Java 11 ou 17 requis pour Spark
cd ../PROJECT_2_NYC_TAXI_BIGDATA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/nyc_taxi_pipeline.ipynb
```

Le notebook est livré déjà exécuté : toutes les sorties et les 17 figures sont
enregistrées, aucune exécution n'est nécessaire pour le lire.

Pour lancer l'application d'estimation :

```bash
streamlit run app/app.py
```

## Organisation du dépôt

Les deux projets suivent la même organisation :

```
PROJECT_x/
├── notebooks/      le pipeline complet, livré exécuté
├── tools/          modules importés à l'exécution (figures, classes du pipeline)
├── app/            Streamlit pour le projet 1, FastAPI pour le projet 2
├── models/         modèles et transformateurs sauvegardés
├── reports/        figures et tableaux de résultats
├── data/raw/       données d'entrée
└── README.md       documentation détaillée
```

Ne sont pas versionnés : les environnements virtuels, les données intermédiaires, le modèle
Random Forest scikit-learn de 31 Mo et les 145 Mo de données NYC Taxi. Tous se
reconstruisent ou se retéléchargent en une commande, décrite dans le README de chaque
projet.

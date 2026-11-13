"""Genere le README du projet 2 a partir des resultats reels du notebook."""
import json
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent


def main():
    comp = pd.read_csv(RACINE / "reports" / "comparaison_modeles.csv")
    cfg = json.loads((RACINE / "models" / "config.json").read_text())
    meilleur = comp.iloc[0]

    tableau = ["| Modèle | RMSE | MAE | R² | RMSLE | MAPE | Entraînement |",
               "|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        tableau.append(
            f"| {r.modele} | {r.RMSE_s:.0f} s | {r.MAE_s:.0f} s | {r.R2:.4f} "
            f"| {r.RMSLE:.4f} | {r.MAPE_pct:.1f} % | {r.entrainement_s:.0f} s |")
    tableau = "\n".join(tableau)

    texte = f"""# Projet 2 : durée des trajets en taxi à New York

Traitement distribué avec Hadoop, HDFS et Apache Spark, inspiré du notebook Kaggle
[`headsortails/nyc-taxi-eda-update-the-fast-the-curious`](https://www.kaggle.com/code/headsortails/nyc-taxi-eda-update-the-fast-the-curious).

## Problématique

Estimer la durée d'une course à partir des seules informations connues **au moment de la
prise en charge**. Une application de réservation doit annoncer une durée avant le départ.

Cette contrainte exclut toute variable produite pendant ou après le trajet : distance au
compteur, prix, pourboire, mode de paiement, heure d'arrivée. C'est le point de rigueur
central du projet.

## Données

| | |
|---|---|
| Source | Données ouvertes NYC Taxi and Limousine Commission |
| Période | Janvier 2016 |
| Volume | 145 Mo en Parquet, {cfg['n_entrainement'] + cfg['n_test']:,} courses retenues après nettoyage |
| Entraînement | {cfg['n_entrainement']:,} courses, jours 1 à 24 |
| Test | {cfg['n_test']:,} courses, jours 25 à 31 |

Le découpage est **chronologique** et non aléatoire : on prédit le futur à partir du passé,
comme en production.

La TLC ayant republié les données de 2016 avec des identifiants de zone à la place des
coordonnées GPS, la distance est estimée par la **médiane historique de chaque itinéraire**,
calculée sur le jeu d'entraînement seul.

## Pipeline

```
BIG DATA → HADOOP/SPARK → TRANSFORMATION → MACHINE LEARNING → ÉVALUATION → DÉPLOIEMENT
```

| Étape | Contenu |
|---|---|
| 1. BIG DATA | Les 3V évalués honnêtement, gain du format Parquet mesuré |
| 2. HADOOP / SPARK | HDFS réel, inspection des blocs, Spark SQL, partitions, Parquet partitionné |
| 3. TRANSFORMATION | Nettoyage justifié, découpage temporel, variables sans fuite |
| 4. MACHINE LEARNING | 4 modèles Spark MLlib dans un pipeline reproductible |
| 5. ÉVALUATION | RMSE, MAE, R², RMSLE, MAPE, analyse par catégorie, Spark contre Pandas |
| 6. DÉPLOIEMENT | API FastAPI, test de non-régression |

## Résultats

Mesurés sur la dernière semaine de janvier, postérieure à tout l'entraînement. Métriques en
secondes, après reconversion depuis le logarithme.

{tableau}

**Modèle déployé : {cfg['modele']}**, RMSE {cfg['rmse_test_s']} s, MAPE
{cfg['mape_test_pct']} %, R² {cfg['r2_test']:.3f}.

## Technologies

Python 3.10, Hadoop 3.4.0, HDFS, Apache Spark 3.5.3, PySpark, Spark SQL, Spark MLlib,
Matplotlib, FastAPI.

## Installation

Java 11 ou 17 est requis. Spark 3.5 ne fonctionne pas avec Java 21.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Démarrer HDFS et charger les données

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
$HADOOP_HOME/sbin/start-dfs.sh

curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-01.parquet
curl -O https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
hdfs dfs -mkdir -p /nyc_taxi/raw
hdfs dfs -put yellow_tripdata_2016-01.parquet taxi_zone_lookup.csv /nyc_taxi/raw/
```

## Exécution

```bash
jupyter lab notebooks/nyc_taxi_pipeline.ipynb      # notebook livré déjà exécuté
uvicorn app.api:api --port 8000                    # API, documentation sur /docs
```

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/predire -H "Content-Type: application/json" \\
  -d '{{"pickup_datetime":"2016-01-28 18:30:00","PULocationID":237,"DOLocationID":132}}'
```

## Limites

- **Un seul mois**, contenant le Nouvel An et la tempête de neige des 22 et 23 janvier 2016.
- **Cluster mono-nœud** : HDFS et Spark sont réels, mais tout tourne sur une machine. Les
  mécanismes sont ceux d'un cluster, pas les performances.
- **Pas de météo ni de trafic temps réel**, pourtant déterminants directs de la durée.
- **Pas de coordonnées GPS** dans la version republiée de 2016.
- **Latence de l'API de l'ordre de la seconde**, due à Spark. Un export ONNX serait
  nécessaire pour du temps réel.

## Perspectives

Traiter les 15 années disponibles, enrichir avec la météo horaire, encoder les itinéraires
par leur durée médiane en validation croisée hors échantillon, exporter le modèle en ONNX,
passer à Spark Structured Streaming.
"""
    (RACINE / "README.md").write_text(texte, encoding="utf-8")
    print("README.md genere depuis les resultats reels.")


if __name__ == "__main__":
    main()

"""Section 1 : BIG DATA."""
from nbgen import md, code


def cells():
    return [

md("""
# Prédiction de la durée d'un trajet en taxi à New York

**Module** : Fondements du Big Data, Projet 2 (Big Data distribué)
**Cible** : `trip_duration`, régression supervisée
**Notebook de référence** : [`headsortails/nyc-taxi-eda-update-the-fast-the-curious`](https://www.kaggle.com/code/headsortails/nyc-taxi-eda-update-the-fast-the-curious)
**Technologies** : Hadoop, HDFS, Apache Spark, PySpark, Spark SQL, Spark MLlib, FastAPI

| # | Étape | Contenu |
|---|-------|---------|
| [1](#etape-1) | BIG DATA | Contexte, les 3V, formats, justification de Spark |
| [2](#etape-2) | HADOOP / SPARK | HDFS réel, SparkSession, partitions, Spark SQL, Parquet |
| [3](#etape-3) | TRANSFORMATION | Nettoyage, GPS aberrants, distance, variables temporelles |
| [4](#etape-4) | MACHINE LEARNING | Spark MLlib, 4 modèles de régression |
| [5](#etape-5) | ÉVALUATION | RMSE, MAE, R², Spark contre Pandas |
| [6](#etape-6) | DÉPLOIEMENT | Modèle sauvegardé et API FastAPI |

**Le notebook de référence est écrit en R Markdown** et porte essentiellement sur
l'exploration. Nous n'exécutons pas de code R : nous reprenons ses analyses pertinentes en
PySpark, et nous ajoutons la partie modélisation qu'il ne couvre pas.

<a id="etape-1"></a>
# Étape 1 : BIG DATA

**Problématique** : estimer la durée d'un trajet à partir des seules informations connues
**au moment de la prise en charge**. Une application de réservation doit annoncer une durée
avant que le trajet ne commence.

**Contrainte méthodologique qui en découle** : toute variable produite *pendant* ou *après*
le trajet est interdite. Cela exclut la distance au compteur, le prix, le pourboire, le mode
de paiement, et bien sûr l'heure d'arrivée. Nous y revenons à l'étape 3.
"""),

code("""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJET = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_RAW = PROJET / "data" / "raw"
DATA_PROC = PROJET / "data" / "processed"
MODELS = PROJET / "models"
REPORTS = PROJET / "reports"
FIGURES = REPORTS / "figures"
APP = PROJET / "app"
for d in (DATA_PROC, MODELS, REPORTS, FIGURES, APP):
    d.mkdir(parents=True, exist_ok=True)

GRAINE = 42
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
HADOOP = Path("/home/ssl/hadoop-3.4.0")
os.environ["PATH"] = f"{HADOOP / 'bin'}:{os.environ['PATH']}"

sys.path.insert(0, str(PROJET / "tools"))
import viz
viz.appliquer_style()

FICHIER = "yellow_tripdata_2016-01.parquet"
CHEMIN_LOCAL = DATA_RAW / FICHIER
CHEMIN_HDFS = f"hdfs://localhost:9000/nyc_taxi/raw/{FICHIER}"


def hdfs(commande):
    \"\"\"Execute une commande HDFS et renvoie sa sortie texte.\"\"\"
    r = subprocess.run(f"{HADOOP}/bin/{commande}", shell=True,
                       capture_output=True, text=True, env=os.environ)
    return (r.stdout + r.stderr).strip()


print(f"Projet : {PROJET}")
print(f"Java   : {os.environ['JAVA_HOME']}")
"""),

md("""
## 1.1 Le dataset

**Source** : données ouvertes de la *NYC Taxi and Limousine Commission*, le régulateur des
taxis new-yorkais. Chaque course de taxi jaune y est enregistrée. Nous utilisons
**janvier 2016**, le même mois et la même source que le jeu Kaggle du notebook de référence,
qui en est un échantillon.

**Pourquoi la source officielle plutôt que l'extrait Kaggle** : 10,9 millions de courses
contre 1,46 million, soit 7 fois plus. Pour un projet dont le sujet est le traitement
distribué, travailler sur le jeu complet rend la démonstration plus honnête.
"""),

code("""
taille_octets = CHEMIN_LOCAL.stat().st_size
apercu = pd.read_parquet(CHEMIN_LOCAL, columns=["tpep_pickup_datetime"])
n_lignes = len(apercu)
del apercu

schema = pd.read_parquet(CHEMIN_LOCAL).head(3)
print(f"Fichier       : {FICHIER}")
print(f"Taille        : {taille_octets / 1024**2:.1f} Mo (Parquet compresse)")
print(f"Lignes        : {n_lignes:,}".replace(",", " "))
print(f"Colonnes      : {len(schema.columns)}")
print(f"Octets/ligne  : {taille_octets / n_lignes:.1f}")
display(schema)
"""),

md("""
## 1.2 Les trois V

| V | Ce que dit ce jeu | Verdict honnête |
|---|---|---|
| **Volume** | 145 Mo compressés, 10,9 M lignes pour un seul mois. Une année complète ferait environ 1,7 Go, et la série 2009-2024 dépasse 50 Go | **Modéré**. Un seul mois tient en mémoire sur une machine correcte |
| **Vélocité** | Environ 350 000 courses par jour, soit 4 par seconde en moyenne et davantage aux heures de pointe. Publication mensuelle en lot | **Faible en l'état**. Le flux temps réel existe chez l'opérateur, pas dans ce fichier |
| **Variété** | Une seule table, schéma stable, types homogènes | **Faible**. Données tabulaires très régulières |

### Ce jeu n'est pas du Big Data au sens industriel

Il faut le dire clairement plutôt que de gonfler l'argumentaire. 145 Mo tiennent dans la
mémoire d'un téléphone. Pandas charge ce fichier en quelques secondes. Aucune des trois
dimensions n'atteint le seuil qui rend un cluster *nécessaire*.

**Son intérêt est pédagogique, et il est réel.** Le jeu est assez gros pour que les
différences de comportement deviennent mesurables : partitionnement, évaluation paresseuse,
coût d'un `shuffle`, bénéfice du format colonne. Ce sont exactement les mécanismes qui
décident de la viabilité d'un traitement sur 50 Go. On les apprend ici à une échelle où
chaque exécution prend des secondes et non des heures.

### Ce qui justifie quand même Spark ici

1. **Le passage à l'échelle est gratuit.** Le code écrit pour ce mois traite les 15 années
   sans une ligne de changement, en ajoutant des machines.
2. **La tolérance aux pannes.** Spark recalcule une partition perdue à partir du lignage.
   Sur un traitement de plusieurs heures, c'est ce qui évite de tout reprendre.
3. **Le modèle de programmation.** Spark SQL décrit *ce que* l'on veut, pas *comment* le
   calculer. L'optimiseur Catalyst s'en charge, et son plan change selon le volume.
"""),

md("""
## 1.3 Pourquoi le format Parquet

Le jeu Kaggle est distribué en CSV, la TLC publie en Parquet. La différence n'est pas
cosmétique : elle conditionne la performance de tout le pipeline.

| | CSV | Parquet |
|---|---|---|
| Organisation | Par ligne | **Par colonne** |
| Compression | Aucune | Snappy, par colonne |
| Typage | Tout en texte, à convertir | Types conservés |
| Lecture partielle | Impossible, il faut tout lire | **Seules les colonnes utiles sont lues** |
| Statistiques | Aucune | Min, max et comptages par bloc |

Le dernier point est décisif. Un filtre `WHERE passenger_count > 6` laisse Spark écarter des
blocs entiers sans les décompresser, grâce aux statistiques stockées dans le fichier. On
appelle cela le *predicate pushdown*. Mesurons.
"""),

code("""
CHEMIN_CSV = DATA_RAW / "echantillon.csv"

# Echantillon : convertir 10,9 M de lignes en CSV produirait environ 1,8 Go
echantillon = pd.read_parquet(CHEMIN_LOCAL).head(500_000)
if not CHEMIN_CSV.exists():
    echantillon.to_csv(CHEMIN_CSV, index=False)
echantillon.to_parquet(DATA_RAW / "echantillon.parquet", index=False)

taille_csv = CHEMIN_CSV.stat().st_size
taille_pq = (DATA_RAW / "echantillon.parquet").stat().st_size

def chronometre(fonction, repetitions=3):
    durees = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        fonction()
        durees.append(time.perf_counter() - debut)
    return min(durees)

t_csv = chronometre(lambda: pd.read_csv(CHEMIN_CSV))
t_pq = chronometre(lambda: pd.read_parquet(DATA_RAW / "echantillon.parquet"))
t_pq_1col = chronometre(
    lambda: pd.read_parquet(DATA_RAW / "echantillon.parquet", columns=["trip_distance"]))

display(pd.DataFrame({
    "taille_Mo": [taille_csv / 1024**2, taille_pq / 1024**2, taille_pq / 1024**2],
    "lecture_s": [t_csv, t_pq, t_pq_1col],
}, index=["CSV complet", "Parquet complet", "Parquet, 1 colonne"]).round(3))

print(f"Compression : x{taille_csv / taille_pq:.1f}")
print(f"Lecture complete : x{t_csv / t_pq:.1f} plus rapide")
print(f"Lecture d'une seule colonne : x{t_csv / t_pq_1col:.1f} plus rapide que le CSV")
"""),

md("""
## 1.4 Problématiques de stockage et de traitement

**Stockage.** Un fichier de 145 Mo tient sur n'importe quel disque. Quinze années de
données, soit plus de 50 Go, posent trois problèmes qu'un système de fichiers classique ne
résout pas : la panne disque fait tout perdre, la lecture est limitée par un seul bras de
lecture, et aucune machine du commerce ne garde cela en mémoire. **HDFS** répond aux trois
en découpant les fichiers en blocs répliqués sur plusieurs machines.

**Traitement.** Un agrégat sur 50 Go en Pandas exige 50 Go de mémoire vive, plus les copies
intermédiaires. **Spark** traite les données par partitions, en mémoire quand elle suffit,
sur disque quand elle manque, et répartit le calcul sur plusieurs cœurs ou machines.

**Ce que nous faisons réellement dans ce notebook.** Un cluster mono-nœud : HDFS tourne avec
un NameNode et un DataNode sur cette seule machine, Spark en mode `local[*]` utilise les
cœurs du processeur. **Ce n'est pas du calcul distribué sur plusieurs machines**, et nous ne
prétendrons pas le contraire. C'est en revanche la même API, le même code, les mêmes
mécanismes de partitionnement : le passage à un vrai cluster ne demanderait que de changer
l'URL du maître.
"""),
    ]

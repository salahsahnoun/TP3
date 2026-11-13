"""Chaine de prediction Spark pour le deploiement.

Seul endroit ou la construction des variables de production est ecrite. Le
notebook et l'API FastAPI importent tous les deux ce module, ce qui garantit
qu'une course soumise a l'API subit exactement le meme traitement qu'une course
du jeu d'entrainement.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")

RACINE = Path(__file__).resolve().parent.parent
MODELS = RACINE / "models"


class PredicteurTrajet:
    """Charge la SparkSession, le pipeline et les tables de reference."""

    def __init__(self, dossier=MODELS, master="local[2]"):
        from pyspark.sql import SparkSession
        from pyspark.ml import PipelineModel

        self.dossier = Path(dossier)
        self.config = json.loads((self.dossier / "config.json").read_text())

        self.spark = (SparkSession.builder
                      .appName("API duree de trajet")
                      .master(master)
                      .config("spark.driver.memory", "2g")
                      .config("spark.sql.shuffle.partitions", "4")
                      .getOrCreate())
        self.spark.sparkContext.setLogLevel("ERROR")

        # Prefixe file:// obligatoire : si Hadoop est configure, le systeme de
        # fichiers par defaut de Spark est HDFS, pas le disque local.
        def local(p):
            return "file://" + str(p)

        self.modele = PipelineModel.load(local(self.dossier / "pipeline_modele"))
        self.distances = self.spark.read.parquet(
            local(self.dossier / "distances_itineraire.parquet"))
        self.zones = self.spark.read.parquet(local(self.dossier / "zones.parquet"))
        self.distance_defaut = float(self.config["distance_globale"])

    def construire(self, courses):
        """courses : liste de dictionnaires decrivant des prises en charge."""
        from pyspark.sql import functions as F

        df = self.spark.createDataFrame([
            {
                "tpep_pickup_datetime": c["pickup_datetime"],
                "PULocationID": int(c["PULocationID"]),
                "DOLocationID": int(c["DOLocationID"]),
                "passenger_count": int(c.get("passenger_count", 1)),
                "RatecodeID": int(c.get("RatecodeID", 1)),
                "VendorID": int(c.get("VendorID", 1)),
                "store_and_fwd_flag": str(c.get("store_and_fwd_flag", "N")),
            } for c in courses
        ]).withColumn("tpep_pickup_datetime",
                      F.to_timestamp("tpep_pickup_datetime"))

        d = (df
             .withColumn("heure", F.hour("tpep_pickup_datetime"))
             .withColumn("jour_semaine", F.dayofweek("tpep_pickup_datetime"))
             .withColumn("minute_du_jour",
                         F.hour("tpep_pickup_datetime") * 60
                         + F.minute("tpep_pickup_datetime"))
             .withColumn("est_weekend",
                         F.when(F.col("jour_semaine").isin(1, 7), 1).otherwise(0))
             .withColumn("est_heure_pointe",
                         F.when(F.col("heure").between(7, 9)
                                | F.col("heure").between(16, 19), 1).otherwise(0))
             .withColumn("est_nuit",
                         F.when((F.col("heure") >= 22) | (F.col("heure") <= 5), 1)
                         .otherwise(0))
             .withColumn("meme_zone",
                         F.when(F.col("PULocationID") == F.col("DOLocationID"), 1)
                         .otherwise(0)))

        d = (d.join(F.broadcast(self.distances),
                    ["PULocationID", "DOLocationID"], "left")
             .withColumn("distance_itineraire",
                         F.coalesce(F.col("distance_itineraire"),
                                    F.lit(self.distance_defaut)))
             .withColumn("itineraire_connu",
                         F.when(F.col("nb_observations").isNotNull(), 1).otherwise(0))
             .drop("nb_observations"))

        arr = self.zones.select("LocationID", "Borough")
        d = (d.join(F.broadcast(arr.withColumnRenamed("Borough", "arr_depart")),
                    d.PULocationID == F.col("LocationID"), "left").drop("LocationID")
             .join(F.broadcast(arr.withColumnRenamed("Borough", "arr_arrivee")),
                   F.col("DOLocationID") == F.col("LocationID"), "left").drop("LocationID")
             .fillna({"arr_depart": "Inconnu", "arr_arrivee": "Inconnu"}))
        return d

    def predire(self, courses):
        """Renvoie la liste des durees estimees, en secondes."""
        from pyspark.sql import functions as F
        sortie = (self.modele.transform(self.construire(courses))
                  .withColumn("duree_s", F.expm1("prediction")))
        return [r["duree_s"] for r in sortie.select("duree_s").collect()]


_PREDICTEUR = None


def obtenir_predicteur():
    global _PREDICTEUR
    if _PREDICTEUR is None:
        _PREDICTEUR = PredicteurTrajet()
    return _PREDICTEUR

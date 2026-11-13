"""Section 6 : DEPLOIEMENT."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-6"></a>
# Étape 6 : DÉPLOIEMENT

## 6.1 Ce qu'il faut sauvegarder

Un modèle seul ne suffit pas. Une course soumise à l'API doit subir **exactement** les mêmes
transformations qu'une course d'entraînement, sinon les prédictions sont fausses sans
qu'aucune erreur ne soit levée.

| Artefact | Rôle |
|---|---|
| `pipeline_modele/` | `PipelineModel` complet : encodeurs ajustés **et** modèle |
| `distances_itineraire.parquet` | Distance historique par couple de zones, apprise sur le train |
| `zones.parquet` | Table des arrondissements |
| `config.json` | Médiane de repli, nom du modèle, métriques affichées par l'API |

Le `PipelineModel` est l'élément clé : il embarque les `StringIndexer` et `OneHotEncoder`
ajustés. Recharger ce seul objet garantit que l'encodage de production est identique à celui
de l'entraînement.
"""),

code("""
import json
import shutil

# ATTENTION : fs.defaultFS vaut hdfs://localhost:9000, donc un chemin sans
# prefixe envoie l'ecriture dans HDFS. Le prefixe file:// est obligatoire pour
# que les artefacts atterrissent sur le disque local, seul endroit ou l'API et
# le depot Git peuvent les lire.
def local(chemin):
    return "file://" + str(chemin)

CHEMIN_MODELE = MODELS / "pipeline_modele"
if CHEMIN_MODELE.exists():
    shutil.rmtree(CHEMIN_MODELE)

# Le modele seul ne suffit pas : on le recombine avec l'encodage ajuste en 4.1
pipeline_complet = PipelineModel(stages=prep_modele.stages + [entraines[MEILLEUR]])
pipeline_complet.write().overwrite().save(local(CHEMIN_MODELE))

(distances_itineraire.coalesce(1).write.mode("overwrite")
 .parquet(local(MODELS / "distances_itineraire.parquet")))
(zones.coalesce(1).write.mode("overwrite")
 .parquet(local(MODELS / "zones.parquet")))

ligne = comparaison[comparaison.modele == MEILLEUR].iloc[0]
config = {
    "modele": MEILLEUR,
    "n_entrainement": int(n_tr),
    "n_test": int(n_te),
    "distance_globale": float(distance_globale),
    "rmse_test_s": int(ligne.RMSE_s),
    "mae_test_s": int(ligne.MAE_s),
    "r2_test": float(ligne.R2),
    "mape_test_pct": float(ligne.MAPE_pct),
    "periode": "janvier 2016",
    "graine": GRAINE,
}
(MODELS / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

taille = sum(f.stat().st_size for f in MODELS.rglob("*") if f.is_file())
print(f"Artefacts sauvegardes : {taille / 1024**2:.1f} Mo")
for p in sorted(MODELS.iterdir()):
    print(f"  {p.name}")
"""),

md("""
## 6.2 Test de non-régression

Le test qui compte : une course passée par la chaîne de déploiement doit recevoir la même
prédiction que dans ce notebook. La tolérance est d'une seconde, pour absorber les arrondis
de sérialisation.
"""),

code("""
sys.path.insert(0, str(APP))
from prediction_spark import PredicteurTrajet

# Le predicteur reutilise la SparkSession active du notebook
predicteur = PredicteurTrajet.__new__(PredicteurTrajet)
predicteur.dossier = MODELS
predicteur.config = json.loads((MODELS / "config.json").read_text())
predicteur.spark = spark
from pyspark.ml import PipelineModel
predicteur.modele = PipelineModel.load(local(CHEMIN_MODELE))
predicteur.distances = spark.read.parquet(local(MODELS / "distances_itineraire.parquet"))
predicteur.zones = spark.read.parquet(local(MODELS / "zones.parquet"))
predicteur.distance_defaut = float(distance_globale)

# Dix courses reelles du jeu de test
echantillon = (te_ml.select("tpep_pickup_datetime", "PULocationID", "DOLocationID",
                            "passenger_count", "RatecodeID", "VendorID",
                            "store_and_fwd_flag", "trip_duration")
               .limit(10).toPandas())

courses = [{"pickup_datetime": str(r.tpep_pickup_datetime),
            "PULocationID": int(r.PULocationID), "DOLocationID": int(r.DOLocationID),
            "passenger_count": int(r.passenger_count), "RatecodeID": int(r.RatecodeID),
            "VendorID": int(r.VendorID), "store_and_fwd_flag": r.store_and_fwd_flag}
           for r in echantillon.itertuples()]

via_api = predicteur.predire(courses)

ids = [(int(r.PULocationID), int(r.DOLocationID), str(r.tpep_pickup_datetime))
       for r in echantillon.itertuples()]
notebook_pred = (predictions_test[MEILLEUR]
                 .withColumn("pred_s", F.expm1("prediction"))
                 .select("tpep_pickup_datetime", "PULocationID", "DOLocationID", "pred_s")
                 .toPandas())

verif = echantillon[["PULocationID", "DOLocationID", "trip_duration"]].copy()
verif["reel_s"] = echantillon.trip_duration
verif["deploiement_s"] = np.round(via_api, 1)
fusion = notebook_pred.merge(
    echantillon[["tpep_pickup_datetime", "PULocationID", "DOLocationID"]],
    on=["tpep_pickup_datetime", "PULocationID", "DOLocationID"]).drop_duplicates(
    subset=["tpep_pickup_datetime", "PULocationID", "DOLocationID"])
verif["notebook_s"] = np.round(fusion.pred_s.values[:len(verif)], 1)
verif["ecart_s"] = (verif.deploiement_s - verif.notebook_s).round(2)
display(verif[["PULocationID", "DOLocationID", "reel_s",
               "notebook_s", "deploiement_s", "ecart_s"]])

ecart_max = float(verif.ecart_s.abs().max())
print(f"Ecart maximal : {ecart_max:.3f} s")
assert ecart_max < 1.0, "La chaine de deploiement ne reproduit pas le notebook."
print("[OK] La chaine de deploiement reproduit le notebook.")
"""),

code("""
# Controle de coherence : la duree doit croitre avec l'eloignement,
# et etre plus longue aux heures de pointe qu'en pleine nuit
scenarios = {
    "Meme zone, nuit": dict(pickup_datetime="2016-01-28 03:00:00",
                            PULocationID=237, DOLocationID=237),
    "Manhattan court, nuit": dict(pickup_datetime="2016-01-28 03:00:00",
                                  PULocationID=237, DOLocationID=161),
    "Manhattan court, heure de pointe": dict(pickup_datetime="2016-01-28 18:00:00",
                                             PULocationID=237, DOLocationID=161),
    "Manhattan vers aeroport JFK": dict(pickup_datetime="2016-01-28 18:00:00",
                                        PULocationID=237, DOLocationID=132),
}
estimations = predicteur.predire(list(scenarios.values()))
for nom, duree in zip(scenarios, estimations):
    print(f"  {nom:36s} {duree / 60:5.1f} min")

assert estimations[2] > estimations[1], "L'heure de pointe devrait allonger le trajet."
assert estimations[3] > estimations[2], "JFK devrait etre plus long qu'un trajet interne."
print("\\n[OK] Les estimations suivent la logique attendue.")
"""),

md("""
## 6.3 L'API FastAPI

FastAPI plutôt que Streamlit pour ce projet : la destination d'une estimation de durée est
une application de réservation, c'est-à-dire **un autre programme**, pas un humain devant un
formulaire. L'API renvoie du JSON et publie sa documentation interactive sur `/docs`.

| Route | Rôle |
|---|---|
| `GET /` | Identité du service et métriques du modèle |
| `GET /sante` | Vérification de disponibilité |
| `POST /predire` | Estimation pour une course |
| `GET /docs` | Documentation interactive générée automatiquement |

**Une limite à connaître** : chaque prédiction passe par Spark, dont la latence est de
l'ordre de la seconde. C'est inadapté à un service temps réel à fort trafic. En production
on exporterait le modèle vers un format léger, par exemple ONNX ou MLeap, pour servir en
quelques millisecondes. Nous conservons Spark ici parce que le sujet porte sur l'écosystème
Hadoop et Spark, et parce que cela garantit l'identité stricte entre entraînement et
inférence.
"""),

code("""
# Demarrage reel du serveur, pour prouver que l'API fonctionne
import subprocess
import urllib.request

serveur = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.api:api",
     "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
    cwd=str(PROJET), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    env={**os.environ, "PYTHONPATH": str(PROJET)})

reponse_racine, reponse_predire, erreur = None, None, None
try:
    for _ in range(90):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/", timeout=3) as r:
                reponse_racine = json.loads(r.read())
                break
        except Exception as e:
            erreur = e
            if serveur.poll() is not None:
                break

    if reponse_racine:
        requete = urllib.request.Request(
            "http://127.0.0.1:8000/predire",
            data=json.dumps({"pickup_datetime": "2016-01-28 18:30:00",
                             "PULocationID": 237, "DOLocationID": 132,
                             "passenger_count": 2}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(requete, timeout=60) as r:
            reponse_predire = json.loads(r.read())
finally:
    serveur.terminate()
    try:
        serveur.wait(timeout=15)
    except subprocess.TimeoutExpired:
        serveur.kill()

print("GET / :"); print(json.dumps(reponse_racine, indent=2, ensure_ascii=False))
print("\\nPOST /predire :"); print(json.dumps(reponse_predire, indent=2, ensure_ascii=False))
assert reponse_predire and "duree_estimee_s" in reponse_predire, f"API injoignable ({erreur})"
print("\\n[OK] L'API demarre, repond et renvoie une estimation. Serveur arrete.")
"""),

md("""
## 6.4 Bilan du projet

### Ce qui a été réalisé

| Étape | Livrable |
|---|---|
| BIG DATA | Les 3V évalués honnêtement, gain du format Parquet mesuré |
| HADOOP / SPARK | HDFS réel mono-nœud, blocs inspectés, Spark SQL, partitions, Parquet partitionné |
| TRANSFORMATION | Nettoyage justifié, découpage temporel, variables sans fuite |
| MACHINE LEARNING | 4 modèles MLlib dans un pipeline reproductible |
| ÉVALUATION | RMSE, MAE, R², RMSLE, MAPE, analyse par catégorie, Spark contre Pandas |
| DÉPLOIEMENT | API FastAPI testée, non-régression vérifiée |

### Limites

- **Un seul mois de données**, contenant deux anomalies majeures : le Nouvel An et la
  tempête de neige des 22 et 23 janvier 2016.
- **Cluster mono-nœud.** HDFS et Spark sont réels, mais tout tourne sur une machine. Les
  mécanismes sont les mêmes, les performances ne sont pas celles d'un cluster.
- **Pas de données météo ni de trafic temps réel**, pourtant déterminants directs.
- **Pas de coordonnées GPS** dans la version republiée de 2016 : la distance est un agrégat
  historique par itinéraire, non une distance réelle.
- **Latence de l'API de l'ordre de la seconde**, due à Spark.

### Perspectives

- Traiter les 15 années disponibles pour mesurer le vrai passage à l'échelle.
- Enrichir avec la météo horaire, librement disponible, et les incidents de circulation.
- Encoder les itinéraires par la durée médiane historique, en validation croisée hors
  échantillon pour éviter la fuite.
- Exporter le modèle en ONNX pour une inférence en millisecondes.
- Passer à Spark Structured Streaming pour traiter les courses au fil de l'eau.
"""),
    ]

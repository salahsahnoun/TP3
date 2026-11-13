"""Section 4 : MACHINE LEARNING avec Spark MLlib."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-4"></a>
# Étape 4 : MACHINE LEARNING

## 4.1 Le pipeline Spark MLlib

Un `Pipeline` MLlib enchaîne des étapes typées, comme scikit-learn :

| Étape | Rôle |
|---|---|
| `StringIndexer` | Texte vers entier. `handleInvalid="keep"` range les modalités inédites dans une catégorie dédiée |
| `OneHotEncoder` | Entier vers vecteur creux, pour ne pas suggérer d'ordre entre catégories nominales |
| `VectorAssembler` | Rassemble toutes les variables dans une colonne `features` |
| Modèle | Entraîné sur cette colonne |

**Intérêt pour le déploiement** : un `PipelineModel` contient *à la fois* les encodeurs
ajustés et le modèle. Il se sauvegarde et se recharge d'un bloc, ce qui supprime le risque
que l'encodage de production diverge de celui de l'entraînement.

**Un point d'efficacité qui compte sur 8 millions de lignes.** L'encodage est ajusté **une
seule fois**, puis le résultat est mis en cache et réutilisé par les quatre modèles. Ajuster
le pipeline complet quatre fois reviendrait à refaire quatre fois le même travail
d'indexation. Les modèles sont ensuite recombinés avec l'encodage pour former le
`PipelineModel` que l'étape 6 déploie.

**L'arrondissement plutôt que la zone.** Encoder les 263 zones en *one-hot* produirait plus
de 500 colonnes, coûteux pour les ensembles d'arbres, alors que la géométrie de l'itinéraire
est déjà portée par `distance_itineraire`.
"""),

code("""
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import (LinearRegression, DecisionTreeRegressor,
                                   RandomForestRegressor, GBTRegressor)
from pyspark.ml.evaluation import RegressionEvaluator

arrondissements = zones.select(F.col("LocationID"), F.col("Borough"))


def ajouter_arrondissements(df):
    return (df
            .join(F.broadcast(arrondissements.withColumnRenamed("Borough", "arr_depart")),
                  df.PULocationID == F.col("LocationID"), "left").drop("LocationID")
            .join(F.broadcast(arrondissements.withColumnRenamed("Borough", "arr_arrivee")),
                  F.col("DOLocationID") == F.col("LocationID"), "left").drop("LocationID")
            .fillna({"arr_depart": "Inconnu", "arr_arrivee": "Inconnu"}))


tr_ml = ajouter_arrondissements(tr).cache()
te_ml = ajouter_arrondissements(te).cache()
print(f"Entrainement : {tr_ml.count():,} | Test : {te_ml.count():,}".replace(",", " "))
tr_ml.groupBy("arr_depart").count().orderBy(F.desc("count")).show(7, truncate=False)
"""),

code("""
CATEGORIELLES = ["arr_depart", "arr_arrivee", "store_and_fwd_flag"]
NUMERIQUES = ["heure", "jour_semaine", "minute_du_jour", "est_weekend",
              "est_heure_pointe", "est_nuit", "meme_zone", "passenger_count",
              "distance_itineraire", "itineraire_connu", "RatecodeID", "VendorID"]

indexeurs = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
             for c in CATEGORIELLES]
encodeur = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORIELLES],
                         outputCols=[f"{c}_vec" for c in CATEGORIELLES],
                         handleInvalid="keep")
assembleur = VectorAssembler(
    inputCols=NUMERIQUES + [f"{c}_vec" for c in CATEGORIELLES],
    outputCol="features", handleInvalid="skip")

# Ajustement UNIQUE de l'encodage, sur le jeu d'entrainement
debut = time.perf_counter()
prep_modele = Pipeline(stages=indexeurs + [encodeur, assembleur]).fit(tr_ml)
print(f"Encodage ajuste en {time.perf_counter() - debut:.1f} s")

# Colonnes conservees apres encodage : la cible, les variables, et ce dont
# les etapes 5 et 6 ont besoin pour l'analyse et le test de non-regression
A_GARDER = ["features", "cible_log", "trip_duration", "est_nuit", "est_heure_pointe",
            "tpep_pickup_datetime", "PULocationID", "DOLocationID",
            "passenger_count", "RatecodeID", "VendorID", "store_and_fwd_flag"]

tr_prep = prep_modele.transform(tr_ml).select(*A_GARDER).cache()
te_prep = prep_modele.transform(te_ml).select(*A_GARDER).cache()
print(f"Entrainement prepare : {tr_prep.count():,}".replace(",", " "))
print(f"Test prepare         : {te_prep.count():,}".replace(",", " "))
print(f"Dimension du vecteur features : {tr_prep.first()['features'].size}")
"""),

md("""
## 4.2 Les quatre modèles

| Modèle | Principe | Attendu ici |
|---|---|---|
| **Linear Regression** | Combinaison linéaire des variables | Référence. Ne peut pas représenter l'effet non linéaire de l'heure |
| **Decision Tree** | Découpages successifs | Capture les non-linéarités, mais un arbre seul est instable |
| **Random Forest** | Moyenne d'arbres décorrélés, construits en parallèle | Réduit la variance, robuste |
| **GBT** | Arbres séquentiels corrigeant le résidu précédent | Souvent le plus précis, nettement plus lent |

**Entraînement en logarithme** : la distribution des durées est fortement asymétrique et la
métrique de référence du sujet est une RMSLE. Les prédictions sont reconverties en secondes
à l'étape 5.

**Coût du GBT en distribué** : la forêt construit ses arbres indépendamment, donc en
parallèle. Le boosting les construit en séquence, chaque arbre dépendant du précédent. Sur
un cluster, la forêt passe mieux à l'échelle.
"""),

code('''
EVALUATEUR = RegressionEvaluator(labelCol="cible_log", predictionCol="prediction",
                                 metricName="rmse")
MODELES = {
    "Linear Regression": LinearRegression(
        featuresCol="features", labelCol="cible_log",
        maxIter=50, regParam=0.01, elasticNetParam=0.0),
    "Decision Tree": DecisionTreeRegressor(
        featuresCol="features", labelCol="cible_log",
        maxDepth=10, maxBins=48, seed=GRAINE),
    "Random Forest": RandomForestRegressor(
        featuresCol="features", labelCol="cible_log",
        numTrees=20, maxDepth=10, maxBins=48, subsamplingRate=0.7, seed=GRAINE),
    # 15 iterations avec un pas plus grand plutot que 25 avec un pas plus
    # petit : le boosting est sequentiel, chaque iteration coute une passe
    # complete sur 8 millions de lignes. Le compromis est assume et documente.
    "GBT": GBTRegressor(
        featuresCol="features", labelCol="cible_log",
        maxIter=15, maxDepth=6, stepSize=0.2, maxBins=48, seed=GRAINE),
}

entraines, journal = {}, []
for nom, estimateur in MODELES.items():
    debut = time.perf_counter()
    modele = estimateur.fit(tr_prep)          # encodage deja fait, donc non refait
    duree = time.perf_counter() - debut

    rmse_tr = EVALUATEUR.evaluate(modele.transform(tr_prep))
    rmse_te = EVALUATEUR.evaluate(modele.transform(te_prep))

    entraines[nom] = modele
    journal.append({"modele": nom,
                    "rmse_log_train": round(rmse_tr, 4),
                    "rmse_log_test": round(rmse_te, 4),
                    "duree_entrainement_s": round(duree, 1)})
    print(f"  {nom:18s} RMSE log test = {rmse_te:.4f}   ({duree:.0f} s)")

resume_entrainement = pd.DataFrame(journal).sort_values("rmse_log_test")
display(resume_entrainement)
'''),

md("""
## 4.3 Importance des variables

Les modèles à base d'arbres fournissent une importance par réduction d'impureté. À lire avec
prudence : cette mesure favorise les variables continues, qui offrent mécaniquement plus de
points de découpage que les variables binaires.
"""),

code("""
# featureImportances est un vecteur creux : toArray() avant tout decoupage
importances = entraines["Random Forest"].featureImportances.toArray()

# Les colonnes one-hot occupent plusieurs positions, on regroupe leur importance
exemple = tr_ml.limit(1)
exemple = prep_modele.transform(exemple).first()
tailles = [1] * len(NUMERIQUES) + [exemple[f"{c}_vec"].size for c in CATEGORIELLES]
noms = NUMERIQUES + CATEGORIELLES

position, groupes = 0, {}
for nom, taille in zip(noms, tailles):
    groupes[nom] = float(importances[position:position + taille].sum())
    position += taille

serie = pd.Series(groupes).sort_values()
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.barh(serie.index, 100 * serie.values, color=viz.BLEU, height=0.68)
ax.set_xlabel("Importance (% de la reduction d'impurete)")
ax.grid(True, axis="x"); ax.grid(False, axis="y")
viz.titrer(ax, "Importance des variables, Random Forest",
           "La distance historique de l'itineraire domine")
fig.tight_layout()
viz.sauver(fig, "fig_02_importances")
plt.show()

display(serie.sort_values(ascending=False).head(6).mul(100).round(1).to_frame("importance_%"))
"""),
    ]

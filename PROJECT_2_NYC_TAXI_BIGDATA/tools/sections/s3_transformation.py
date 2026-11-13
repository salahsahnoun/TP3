"""Section 3 : TRANSFORMATION."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-3"></a>
# Étape 3 : TRANSFORMATION

## 3.1 La contrainte qui commande tout : ce qui est connu au départ

La cible se construit depuis deux horodatages :
`trip_duration = tpep_dropoff_datetime - tpep_pickup_datetime`.

Il faut donc trier les colonnes selon leur disponibilité **au moment de la prise en
charge**. C'est le point de rigueur central de ce projet.

| Colonne | Disponible au départ | Décision |
|---|---|---|
| `tpep_pickup_datetime` | Oui | **Utilisée** |
| `PULocationID`, `DOLocationID` | Oui, le client annonce sa destination | **Utilisées** |
| `passenger_count` | Oui | **Utilisée** |
| `RatecodeID`, `store_and_fwd_flag`, `VendorID` | Oui | **Utilisées** |
| `trip_distance` | **Non**, mesurée par le compteur pendant le trajet | **Exclue des variables** |
| `fare_amount`, `tip_amount`, `total_amount`, `payment_type`, péages | **Non**, connus à l'arrivée | **Exclues** |
| `tpep_dropoff_datetime` | **Non**, c'est l'arrivée | **Sert uniquement à construire la cible** |

**Utiliser `trip_distance` serait la fuite classique de ce sujet.** La distance au compteur
est quasi proportionnelle à la durée : un modèle qui l'utilise atteindrait un R² très élevé
et serait inutilisable en production, puisque la valeur n'existe pas au moment où on doit
répondre.

**Notre contournement** : calculer la **distance médiane historique de chaque itinéraire**
(couple zone de départ, zone d'arrivée), **sur les données d'entraînement uniquement**.
C'est l'équivalent d'une estimation de distance par un service de cartographie, parfaitement
disponible au moment de la réservation.

## 3.2 Une différence avec le sujet d'origine

L'énoncé demande de détecter les coordonnées GPS aberrantes. Ce n'est pas réalisable ici :
la TLC a **republié les données de 2016 en remplaçant les latitudes et longitudes par des
identifiants de zone**, par anonymisation. Le contrôle géographique devient donc une
validation des identifiants de zone, et la distance se calcule par agrégat historique plutôt
que par formule de haversine.
"""),

code("""
from pyspark.sql import Window

DUREE_MIN, DUREE_MAX = 60, 7200          # de 1 minute a 2 heures
VITESSE_MIN, VITESSE_MAX = 1.0, 80.0     # miles par heure plausibles
ZONE_MAX = 263                           # 264 et 265 valent "inconnu" dans la TLC

base = (spark.read.parquet(CHEMIN_HDFS)
        .withColumn("trip_duration",
                    F.unix_timestamp("tpep_dropoff_datetime")
                    - F.unix_timestamp("tpep_pickup_datetime")))

n_depart = base.count()
print(f"Lignes brutes : {n_depart:,}".replace(",", " "))
"""),

code("""
# Chaque filtre est compte separement pour justifier les suppressions
controles = {
    "Date hors janvier 2016": ~F.col("tpep_pickup_datetime").between("2016-01-01", "2016-02-01"),
    "Duree hors 1 min a 2 h": ~F.col("trip_duration").between(DUREE_MIN, DUREE_MAX),
    "Passagers hors 1 a 6": ~F.col("passenger_count").between(1, 6),
    "Distance hors 0,1 a 100 miles": ~F.col("trip_distance").between(0.1, 100),
    "Zone de depart invalide": ~F.col("PULocationID").between(1, ZONE_MAX),
    "Zone d'arrivee invalide": ~F.col("DOLocationID").between(1, ZONE_MAX),
}
comptes = base.select([F.sum(c.cast("int")).alias(nom) for nom, c in controles.items()]).first()

display(pd.DataFrame({
    "lignes_concernees": [comptes[nom] for nom in controles],
    "part_pct": [round(100 * comptes[nom] / n_depart, 3) for nom in controles],
}, index=list(controles)))
"""),

code("""
propre = base
for condition in controles.values():
    propre = propre.filter(~condition)

# La vitesse se calcule apres, elle depend de deux colonnes deja filtrees
propre = (propre
          .withColumn("vitesse_mph",
                      F.col("trip_distance") / (F.col("trip_duration") / 3600.0))
          .filter(F.col("vitesse_mph").between(VITESSE_MIN, VITESSE_MAX)))

propre = propre.cache()
n_propre = propre.count()
print(f"Lignes conservees : {n_propre:,}".replace(",", " "))
print(f"Lignes ecartees   : {n_depart - n_propre:,} "
      f"({100 * (n_depart - n_propre) / n_depart:.2f} %)".replace(",", " "))
"""),

md("""
**Le filtre sur la vitesse est le plus utile.** Il élimine des incohérences qu'aucun
contrôle isolé ne détecte : un trajet de 40 miles en 3 minutes passe les filtres de durée et
de distance pris séparément, mais correspond à 800 mph, donc à une erreur de compteur ou de
GPS. Croiser deux colonnes révèle ce qu'une colonne seule ne montre pas.

## 3.3 Découpage temporel

Pour un problème de prévision, un découpage **aléatoire** est optimiste : il laisse le
modèle apprendre sur des courses du mardi après-midi pour prédire d'autres courses du même
mardi après-midi, dans le même embouteillage.

Nous découpons donc **chronologiquement** : les 24 premiers jours pour apprendre, les 7
derniers pour évaluer. C'est exactement la situation réelle, où l'on prédit le futur à
partir du passé.
"""),

code("""
JOUR_BASCULE = 25

propre = propre.withColumn("jour", F.dayofmonth("tpep_pickup_datetime"))
entrainement = propre.filter(F.col("jour") < JOUR_BASCULE)
test = propre.filter(F.col("jour") >= JOUR_BASCULE)

n_tr, n_te = entrainement.count(), test.count()
print(f"Entrainement : {n_tr:,} courses (jours 1 a {JOUR_BASCULE - 1})".replace(",", " "))
print(f"Test         : {n_te:,} courses (jours {JOUR_BASCULE} a 31)".replace(",", " "))
print(f"Repartition  : {100 * n_tr / (n_tr + n_te):.1f} % / {100 * n_te / (n_tr + n_te):.1f} %")

stats = (propre.groupBy(F.col("jour") >= JOUR_BASCULE)
         .agg(F.round(F.avg("trip_duration") / 60, 2).alias("duree_moy_min"),
              F.round(F.expr("percentile_approx(trip_duration, 0.5)") / 60, 2).alias("mediane_min"))
         .withColumnRenamed("(jour >= 25)", "est_test"))
stats.show()
"""),

md("""
## 3.4 Distance historique par itinéraire

L'agrégat est calculé **sur le jeu d'entraînement seul**, puis joint aux deux jeux. Calculer
la médiane sur l'ensemble ferait entrer de l'information de la semaine de test dans le
modèle.

Les itinéraires absents du jeu d'entraînement reçoivent la médiane globale : c'est le cas
d'une destination jamais observée, qui se produira aussi en production.
"""),

code("""
distances_itineraire = (entrainement
                        .groupBy("PULocationID", "DOLocationID")
                        .agg(F.expr("percentile_approx(trip_distance, 0.5)")
                             .alias("distance_itineraire"),
                             F.count("*").alias("nb_observations")))

distance_globale = (entrainement.select(
    F.expr("percentile_approx(trip_distance, 0.5)")).first()[0])

print(f"Itineraires distincts appris : {distances_itineraire.count():,}".replace(",", " "))
print(f"Mediane globale de repli     : {distance_globale:.2f} miles")
distances_itineraire.orderBy(F.desc("nb_observations")).show(5, truncate=False)
"""),

code('''
def construire_variables(df, table_distances, distance_defaut):
    """Ajoute les variables disponibles au moment de la prise en charge."""
    d = (df
         .withColumn("heure", F.hour("tpep_pickup_datetime"))
         .withColumn("jour_semaine", F.dayofweek("tpep_pickup_datetime"))
         .withColumn("minute_du_jour",
                     F.hour("tpep_pickup_datetime") * 60 + F.minute("tpep_pickup_datetime"))
         .withColumn("est_weekend",
                     F.when(F.col("jour_semaine").isin(1, 7), 1).otherwise(0))
         .withColumn("est_heure_pointe",
                     F.when(F.col("heure").between(7, 9)
                            | F.col("heure").between(16, 19), 1).otherwise(0))
         .withColumn("est_nuit",
                     F.when((F.col("heure") >= 22) | (F.col("heure") <= 5), 1).otherwise(0))
         .withColumn("meme_zone",
                     F.when(F.col("PULocationID") == F.col("DOLocationID"), 1).otherwise(0)))

    # Jointure diffusee : la table des itineraires est petite devant les courses
    d = (d.join(F.broadcast(table_distances), ["PULocationID", "DOLocationID"], "left")
         .withColumn("distance_itineraire",
                     F.coalesce(F.col("distance_itineraire"), F.lit(distance_defaut)))
         .withColumn("itineraire_connu",
                     F.when(F.col("nb_observations").isNotNull(), 1).otherwise(0))
         .drop("nb_observations"))

    # La cible est modelisee en logarithme : la metrique de reference de ce
    # sujet est une RMSLE, et la distribution des durees est tres asymetrique
    return d.withColumn("cible_log", F.log1p(F.col("trip_duration")))


tr = construire_variables(entrainement, distances_itineraire, distance_globale).cache()
te = construire_variables(test, distances_itineraire, distance_globale).cache()

VARIABLES_NUM = ["heure", "jour_semaine", "minute_du_jour", "est_weekend",
                 "est_heure_pointe", "est_nuit", "meme_zone", "passenger_count",
                 "distance_itineraire", "itineraire_connu"]
VARIABLES_CAT = ["PULocationID", "DOLocationID", "RatecodeID", "store_and_fwd_flag",
                 "VendorID"]

print(f"Entrainement : {tr.count():,} lignes".replace(",", " "))
print(f"{len(VARIABLES_NUM)} variables numeriques, {len(VARIABLES_CAT)} categorielles")
tr.select(VARIABLES_NUM[:6] + ["distance_itineraire", "trip_duration"]).show(5)
'''),

code("""
# Controle anti-fuite : aucune colonne posterieure au depart ne doit figurer
INTERDITES = {"trip_distance", "fare_amount", "tip_amount", "total_amount",
              "payment_type", "tolls_amount", "extra", "mta_tax", "vitesse_mph",
              "improvement_surcharge", "congestion_surcharge", "airport_fee",
              "tpep_dropoff_datetime", "trip_duration", "cible_log"}
utilisees = set(VARIABLES_NUM) | set(VARIABLES_CAT)
fuite = utilisees & INTERDITES
print(f"Variables utilisees : {len(utilisees)}")
print(f"Variables interdites presentes : {fuite if fuite else 'aucune'}")
assert not fuite, "Une variable posterieure au depart a ete retenue."
print("\\n[OK] Toutes les variables retenues sont connues au moment de la prise en charge.")
"""),

code("""
# Une figure pour le rapport : profil horaire de la duree moyenne
profil = (tr.groupBy("heure")
          .agg((F.avg("trip_duration") / 60).alias("duree_min"),
               F.count("*").alias("nb"))
          .orderBy("heure").toPandas())

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(profil.heure, profil.duree_min, color=viz.BLEU, lw=2.2, marker="o", ms=5)
axes[0].set_xlabel("Heure de prise en charge")
axes[0].set_ylabel("Duree moyenne (min)")
viz.titrer(axes[0], "Duree moyenne selon l'heure",
           "Pic en fin d'apres-midi, creux la nuit")
axes[1].bar(profil.heure, profil.nb / 1000, color=viz.AQUA, width=0.75)
axes[1].set_xlabel("Heure de prise en charge")
axes[1].set_ylabel("Courses (milliers)")
viz.titrer(axes[1], "Volume de courses selon l'heure",
           "Activite minimale vers 5 h du matin")
fig.tight_layout()
viz.sauver(fig, "fig_01_profil_horaire")
plt.show()
"""),

md("""
**Le profil horaire justifie les variables temporelles.** La durée moyenne varie fortement
selon l'heure, avec un maximum en fin d'après-midi et un minimum nocturne. L'écart entre les
deux dépasse largement le bruit : l'heure est donc un prédicteur de premier ordre, et c'est
la congestion qu'elle capture.
"""),
    ]

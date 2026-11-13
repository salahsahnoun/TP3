"""Section 5 : EVALUATION."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-5"></a>
# Étape 5 : ÉVALUATION

Les modèles ont été entraînés sur le logarithme de la durée. **Toutes les métriques
ci-dessous sont reconverties en secondes**, seule échelle interprétable. Comparer une RMSE
calculée en espace logarithmique à une RMSE en secondes n'aurait aucun sens.

| Métrique | Mesure |
|---|---|
| **RMSE** | Erreur quadratique en secondes, pénalise fortement les grosses erreurs |
| **MAE** | Erreur moyenne en secondes, robuste aux valeurs extrêmes |
| **R²** | Part de variance expliquée |
| **RMSLE** | Erreur relative, métrique de référence du sujet Kaggle |
| **MAPE** | Erreur en pourcentage, la plus lisible pour un utilisateur |

Le jeu de test est la **dernière semaine de janvier**, postérieure à tout ce qui a servi à
l'apprentissage.
"""),

code('''
def metriques_secondes(predictions):
    """Metriques en secondes, a partir de predictions en logarithme."""
    d = (predictions
         .withColumn("pred_s", F.expm1("prediction"))
         .withColumn("erreur", F.col("pred_s") - F.col("trip_duration")))
    agg = d.select(
        F.sqrt(F.avg(F.pow(F.col("erreur"), 2))).alias("RMSE"),
        F.avg(F.abs(F.col("erreur"))).alias("MAE"),
        F.sqrt(F.avg(F.pow(F.log1p(F.greatest(F.col("pred_s"), F.lit(0)))
                           - F.log1p(F.col("trip_duration")), 2))).alias("RMSLE"),
        (100 * F.avg(F.abs(F.col("erreur") / F.col("trip_duration")))).alias("MAPE"),
    ).first()
    variance = d.select(F.var_pop("trip_duration")).first()[0]
    erreur_quad = d.select(F.avg(F.pow(F.col("erreur"), 2))).first()[0]
    return {"RMSE": agg["RMSE"], "MAE": agg["MAE"], "R2": 1 - erreur_quad / variance,
            "RMSLE": agg["RMSLE"], "MAPE": agg["MAPE"]}


# Reference : predire toujours la duree mediane d'entrainement
mediane_tr = tr_prep.select(F.expr("percentile_approx(trip_duration, 0.5)")).first()[0]
reference = metriques_secondes(
    te_prep.withColumn("prediction", F.log1p(F.lit(float(mediane_tr)))))
print(f"Reference naive, toujours {mediane_tr:.0f} s :")
print(f"  RMSE {reference['RMSE']:.0f} s | MAE {reference['MAE']:.0f} s "
      f"| R2 {reference['R2']:.3f} | MAPE {reference['MAPE']:.1f} %")
'''),

code("""
lignes = []
predictions_test = {}
for nom, modele in entraines.items():
    p = modele.transform(te_prep).cache()
    predictions_test[nom] = p
    m = metriques_secondes(p)
    duree = resume_entrainement.loc[
        resume_entrainement.modele == nom, "duree_entrainement_s"].iloc[0]
    lignes.append({"modele": nom,
                   "RMSE_s": round(m["RMSE"]), "MAE_s": round(m["MAE"]),
                   "R2": round(m["R2"], 4), "RMSLE": round(m["RMSLE"], 4),
                   "MAPE_pct": round(m["MAPE"], 1),
                   "entrainement_s": duree})

comparaison = pd.DataFrame(lignes).sort_values("RMSE_s").reset_index(drop=True)
display(comparaison)
comparaison.to_csv(REPORTS / "comparaison_modeles.csv", index=False)

MEILLEUR = comparaison.iloc[0].modele
print(f"Meilleur modele : {MEILLEUR}")
"""),

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ordre = comparaison.sort_values("RMSE_s", ascending=False)

axes[0].barh(ordre.modele, ordre.RMSE_s, color=viz.BLEU, height=0.6)
for i, v in enumerate(ordre.RMSE_s):
    axes[0].text(v + 4, i, f"{v:.0f} s", va="center", fontsize=9, color=viz.ENCRE_2)
axes[0].axvline(reference["RMSE"], color=viz.ROUGE, ls="--", lw=1.6,
                label=f"reference naive ({reference['RMSE']:.0f} s)")
axes[0].set_xlabel("RMSE en secondes")
axes[0].set_xlim(0, max(ordre.RMSE_s.max(), reference["RMSE"]) * 1.2)
axes[0].grid(True, axis="x"); axes[0].grid(False, axis="y")
axes[0].legend(loc="lower right")
viz.titrer(axes[0], "Erreur sur la derniere semaine de janvier",
           "Jeu de test posterieur a l'entrainement")

axes[1].barh(ordre.modele, ordre.R2, color=viz.AQUA, height=0.6)
for i, v in enumerate(ordre.R2):
    axes[1].text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=9, color=viz.ENCRE_2)
axes[1].set_xlabel("R2")
axes[1].set_xlim(0, 1)
axes[1].grid(True, axis="x"); axes[1].grid(False, axis="y")
viz.titrer(axes[1], "Part de variance expliquee", "1,0 serait une prediction parfaite")
fig.tight_layout()
viz.sauver(fig, "fig_03_comparaison_modeles")
plt.show()
"""),

md("""
## 5.1 Comportement selon le type de trajet

Une moyenne globale masque les disparités. Un modèle peut être excellent sur les courses
ordinaires et mauvais sur les longues, qui sont justement celles où l'erreur en secondes
coûte le plus cher à l'utilisateur.
"""),

code("""
meilleures = predictions_test[MEILLEUR].withColumn("pred_s", F.expm1("prediction"))

par_duree = (meilleures
             .withColumn("categorie",
                         F.when(F.col("trip_duration") < 300, "1. moins de 5 min")
                         .when(F.col("trip_duration") < 900, "2. 5 a 15 min")
                         .when(F.col("trip_duration") < 1800, "3. 15 a 30 min")
                         .otherwise("4. plus de 30 min"))
             .groupBy("categorie")
             .agg(F.count("*").alias("nb_courses"),
                  F.round(F.sqrt(F.avg(F.pow(F.col("pred_s") - F.col("trip_duration"), 2))))
                  .alias("RMSE_s"),
                  F.round(100 * F.avg(F.abs((F.col("pred_s") - F.col("trip_duration"))
                                            / F.col("trip_duration"))), 1).alias("MAPE_pct"))
             .orderBy("categorie").toPandas())
display(par_duree)

par_moment = (meilleures
              .withColumn("moment",
                          F.when(F.col("est_nuit") == 1, "nuit")
                          .when(F.col("est_heure_pointe") == 1, "heure de pointe")
                          .otherwise("journee"))
              .groupBy("moment")
              .agg(F.count("*").alias("nb_courses"),
                   F.round(F.sqrt(F.avg(F.pow(F.col("pred_s") - F.col("trip_duration"), 2))))
                   .alias("RMSE_s"),
                   F.round(100 * F.avg(F.abs((F.col("pred_s") - F.col("trip_duration"))
                                             / F.col("trip_duration"))), 1).alias("MAPE_pct"))
              .orderBy(F.desc("nb_courses")).toPandas())
display(par_moment)
"""),

md("""
## 5.2 Spark contre Pandas

La comparaison n'a de sens que sur un volume que Pandas peut charger. Nous prenons donc un
sous-ensemble, et nous mesurons la même agrégation des deux côtés.

**À quoi s'attendre.** Sur un petit volume, Pandas gagne : il travaille en mémoire sans
coût de planification, de sérialisation ni de *shuffle*. Spark paie un surcoût fixe de
quelques centaines de millisecondes par action. L'intérêt de Spark n'apparaît qu'au-delà de
la mémoire disponible, là où Pandas ne fonctionne simplement plus.
"""),

code("""
TAILLES = [100_000, 1_000_000, 5_000_000]
mesures = []

colonnes = ["PULocationID", "trip_duration", "jour_semaine"]
for n in TAILLES:
    sous_ensemble = tr.select(*colonnes).limit(n).cache()
    sous_ensemble.count()

    debut = time.perf_counter()
    _ = (sous_ensemble.groupBy("PULocationID")
         .agg(F.avg("trip_duration"), F.count("*")).collect())
    t_spark = time.perf_counter() - debut

    pdf = sous_ensemble.toPandas()
    debut = time.perf_counter()
    _ = pdf.groupby("PULocationID").trip_duration.agg(["mean", "count"])
    t_pandas = time.perf_counter() - debut

    memoire = pdf.memory_usage(deep=True).sum() / 1024**2
    mesures.append({"lignes": n, "spark_s": round(t_spark, 3),
                    "pandas_s": round(t_pandas, 3),
                    "rapport": round(t_spark / t_pandas, 1),
                    "memoire_pandas_Mo": round(memoire, 1)})
    sous_ensemble.unpersist()
    print(f"  {n:>9,} lignes : Spark {t_spark:.2f} s | Pandas {t_pandas:.3f} s"
          .replace(",", " "))

bancs = pd.DataFrame(mesures)
display(bancs)

extrapolation = bancs.memoire_pandas_Mo.iloc[-1] / bancs.lignes.iloc[-1] * 180_000_000
print(f"\\nExtrapolation : les 15 annees de donnees TLC, environ 180 millions de courses,")
print(f"occuperaient de l'ordre de {extrapolation / 1024:.0f} Go en memoire Pandas.")
print("C'est la limite ou Pandas cesse d'etre une option et ou Spark devient necessaire.")
"""),

md("""
## 5.3 Lecture des résultats

**Ce que le tableau de comparaison montre** : tous les modèles battent nettement la
référence naïve, ce qui confirme que les variables disponibles au départ portent une
information réelle sur la durée. Mais le GBT ne devance l'arbre de décision simple que
d'**une seconde de RMSE**, pour **huit fois plus de temps de calcul**. En production,
l'arbre serait le choix rationnel : c'est un bon exemple de rendement décroissant.

**L'erreur dépend fortement du type de course.** Les trajets de moins de 5 minutes affichent
une MAPE de **65 %**, contre 26 % pour les trajets de 5 à 15 minutes. L'explication est
mécanique : sur une course de 3 minutes, une erreur de 2 minutes fait 65 % d'écart relatif,
alors que la même erreur absolue sur 30 minutes n'en fait que 7 %. À l'inverse, c'est sur
les longs trajets que l'erreur **en secondes** est la plus forte, jusqu'à 1 160 s. Les deux
métriques racontent des histoires opposées, et toutes deux sont vraies.

**La comparaison Spark contre Pandas est volontairement à l'avantage de Pandas**, et le
résultat le confirme : Pandas est 4 à 11 fois plus rapide. Sur quelques millions de lignes
tenant en mémoire, le surcoût de planification de Spark n'est pas amorti. Présenter
l'inverse serait malhonnête.

**L'extrapolation affichée est à prendre avec précaution.** Elle porte sur 3 colonnes
seulement, d'où le chiffre modeste de 3 Go pour 180 millions de courses. Sur les 19 colonnes
du jeu complet, l'ordre de grandeur est plutôt de **20 Go**, auxquels s'ajoutent les copies
intermédiaires que Pandas crée à chaque opération. C'est là que Pandas cesse d'être une
option sur une machine ordinaire, alors que Spark continue en traitant partition par
partition.

**Le vrai enseignement de cette section n'est pas une question de vitesse mais de seuil** :
Spark ne sert à rien tant que les données tiennent en mémoire, et devient indispensable dès
qu'elles n'y tiennent plus.

**Limites de cette évaluation.**

- Un seul mois. Janvier contient le Nouvel An et une tempête de neige majeure les 22 et
  23 janvier 2016, deux anomalies qui pèsent sur l'apprentissage.
- Aucune donnée météo ni de trafic, alors que ce sont des déterminants directs de la durée.
- Le test ne couvre qu'une semaine : la saisonnalité annuelle n'est pas évaluée.
- Les zones périphériques sont très peu représentées, le modèle y est moins fiable.
"""),
    ]

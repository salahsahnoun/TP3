"""Section 2 : HADOOP et SPARK."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-2"></a>
# Étape 2 : HADOOP et SPARK

## 2.1 Architecture HDFS

HDFS découpe chaque fichier en **blocs de 128 Mo par défaut**, répartis sur les machines du
cluster et répliqués.

| Composant | Rôle |
|---|---|
| **NameNode** | Garde en mémoire l'arborescence et la carte fichier vers blocs. Il ne stocke aucune donnée. C'est le point unique de défaillance. |
| **DataNode** | Stocke les blocs et répond directement aux clients. Envoie un battement de cœur au NameNode toutes les 3 secondes. |
| **SecondaryNameNode** | Fusionne périodiquement le journal des modifications avec l'image du système de fichiers. Ce n'est **pas** un NameNode de secours, malgré son nom. |

Trois principes expliquent les choix de conception :

1. **Écrire une fois, lire souvent.** Pas de modification en place, seulement de l'ajout.
   Cela simplifie énormément la cohérence.
2. **Déplacer le calcul vers les données.** Transférer 128 Mo coûte plus cher qu'envoyer le
   programme sur la machine qui les détient.
3. **La panne est la norme.** Avec mille machines, il y en a toujours une en panne. D'où la
   réplication, par défaut sur 3 machines.

**Sur notre installation** : un seul nœud, donc `dfs.replication = 1`. Une panne disque
perdrait tout. C'est acceptable pour un travail pédagogique, inacceptable en production.
"""),

code("""
print(hdfs("hdfs dfsadmin -report").split("Live datanodes")[0][:700])
"""),

code("""
print("Contenu du repertoire HDFS :")
print(hdfs("hdfs dfs -ls -h /nyc_taxi/raw"))
print("\\nDecoupage en blocs du fichier principal :")
rapport = hdfs(f"hdfs fsck /nyc_taxi/raw/{FICHIER} -files -blocks")
for ligne in rapport.splitlines():
    if any(m in ligne for m in ["Total size", "Total blocks", "len=", "Status", "replication"]):
        print("  " + ligne.strip()[:110])
"""),

md("""
**Lecture du résultat** : 151 Mo découpés en **2 blocs**, le premier plein à 128 Mo, le
second contenant le reste. Sur un vrai cluster, ces deux blocs vivraient sur deux machines
différentes et seraient lus en parallèle. Ici ils sont sur le même disque : le
parallélisme est celui des cœurs du processeur, pas celui des machines.

## 2.2 MapReduce contre Spark

| | Hadoop MapReduce | Apache Spark |
|---|---|---|
| Unité de calcul | Map puis Reduce, imposés | Graphe de transformations quelconque |
| Entre deux étapes | Écriture sur disque | **En mémoire** |
| Itérations | Relit le disque à chaque tour | Garde le jeu en mémoire |
| Code | Verbeux, Java | DataFrames, SQL, Python |
| Ordre de grandeur | Référence | 10 à 100 fois plus rapide sur l'itératif |

Le gain vient d'un seul point : MapReduce **matérialise sur disque** le résultat de chaque
étape. Un algorithme d'apprentissage qui fait 100 passes sur les données écrit et relit 100
fois. Spark garde la partition en mémoire et ne l'écrit que si nécessaire.

C'est décisif pour l'étape 4 : entraîner un modèle, c'est précisément itérer.

## 2.3 La SparkSession

`local[*]` signifie : un seul processus, autant de threads de calcul que de cœurs
disponibles. Les notions de partition, de *shuffle* et de plan d'exécution restent
identiques à celles d'un cluster, seule l'implantation physique change.
""" ),

code("""
from pyspark.sql import SparkSession, functions as F, types as T

spark = (SparkSession.builder
         .appName("NYC Taxi Trip Duration")
         .master("local[*]")
         .config("spark.driver.memory", "3g")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.adaptive.enabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

sc = spark.sparkContext
print(f"Spark          : {spark.version}")
print(f"Maitre         : {sc.master}")
print(f"Coeurs alloues : {sc.defaultParallelism}")
print(f"Interface web  : http://localhost:4040")
"""),

code("""
debut = time.perf_counter()
courses = spark.read.parquet(CHEMIN_HDFS)
t_lecture = time.perf_counter() - debut

print(f"Declaration de la lecture : {t_lecture:.4f} s")
print("Aucune donnee n'a ete lue : read() est une TRANSFORMATION, donc paresseuse.\\n")

debut = time.perf_counter()
n = courses.count()
print(f"count() : {n:,} lignes en {time.perf_counter() - debut:.2f} s".replace(",", " "))
print(f"Partitions : {courses.rdd.getNumPartitions()}")
"""),

md("""
## 2.4 Transformations et actions

C'est le mécanisme central de Spark, et la source des surprises les plus fréquentes.

| | Exemples | Comportement |
|---|---|---|
| **Transformation** | `select`, `filter`, `withColumn`, `groupBy`, `join` | **Paresseuse**. Spark enregistre l'opération dans un graphe et ne calcule rien. |
| **Action** | `count`, `collect`, `show`, `write` | **Déclenche** le calcul de tout le graphe accumulé. |

L'intérêt n'est pas théorique : comme Spark connaît l'enchaînement complet avant de
l'exécuter, l'optimiseur Catalyst peut réordonner les opérations, pousser les filtres au
plus près de la lecture, et ne lire que les colonnes réellement utilisées.

Vérifions sur une chaîne de trois transformations.
"""),

code("""
chaine = (courses
          .filter(F.col("trip_distance") > 1)
          .select("tpep_pickup_datetime", "PULocationID", "trip_distance")
          .withColumn("heure", F.hour("tpep_pickup_datetime")))

debut = time.perf_counter()
_ = chaine
print(f"Construction de la chaine : {time.perf_counter() - debut:.5f} s (rien n'est calcule)")

debut = time.perf_counter()
resultat = chaine.count()
print(f"Action count()            : {time.perf_counter() - debut:.2f} s -> {resultat:,} lignes"
      .replace(",", " "))

print("\\nPlan d'execution physique retenu par Catalyst :")
chaine.explain(mode="formatted")
"""),

md("""
**Deux optimisations visibles dans le plan**, et elles sont automatiques :

- `PushedFilters: [GreaterThan(trip_distance,1.0)]` : le filtre est **descendu dans le
  lecteur Parquet**. Les blocs dont le maximum de `trip_distance` est inférieur à 1 ne sont
  même pas décompressés.
- `ReadSchema: struct<...>` ne contient que 3 colonnes sur 19 : les 16 autres ne sont jamais
  lues sur le disque.

Nous n'avons demandé ni l'un ni l'autre. C'est la différence entre décrire *ce que* l'on
veut et écrire *comment* le calculer.

## 2.5 Spark SQL

La même donnée est interrogeable en SQL, avec exactement le même optimiseur et les mêmes
performances. Utile pour exprimer une agrégation complexe de façon lisible.
"""),

code("""
courses.createOrReplaceTempView("courses")

spark.sql(\"\"\"
    SELECT  hour(tpep_pickup_datetime)                           AS heure,
            COUNT(*)                                             AS nb_courses,
            ROUND(AVG(trip_distance), 2)                         AS distance_moy_miles,
            ROUND(AVG(unix_timestamp(tpep_dropoff_datetime)
                    - unix_timestamp(tpep_pickup_datetime)) / 60, 1) AS duree_moy_min
    FROM    courses
    WHERE   trip_distance BETWEEN 0.1 AND 100
    GROUP BY heure
    ORDER BY heure
\"\"\").show(24, truncate=False)
"""),

code("""
# Jointure avec la table des zones : elle est petite, Spark la diffuse a tous
# les executeurs (broadcast join) plutot que de redistribuer la grande table.
zones = (spark.read
         .option("header", "true").option("inferSchema", "true")
         .csv("hdfs://localhost:9000/nyc_taxi/raw/taxi_zone_lookup.csv"))
zones.createOrReplaceTempView("zones")
print(f"Table des zones : {zones.count()} lignes")

spark.sql(\"\"\"
    SELECT  z.Borough                 AS arrondissement,
            COUNT(*)                  AS nb_departs,
            ROUND(100 * COUNT(*) / (SELECT COUNT(*) FROM courses), 1) AS part_pct
    FROM    courses c
    JOIN    zones z ON c.PULocationID = z.LocationID
    GROUP BY z.Borough
    ORDER BY nb_departs DESC
\"\"\").show(truncate=False)
"""),

md("""
**Manhattan concentre l'écrasante majorité des départs.** C'est cohérent avec le marché des
taxis jaunes, cantonnés au centre, les autres arrondissements étant desservis par les taxis
verts et les VTC. Conséquence pour la modélisation : les zones périphériques seront peu
représentées, et le modèle y sera moins fiable.

## 2.6 Partitions

La **partition** est l'unité de parallélisme : une partition est traitée par un cœur à la
fois. Trop peu de partitions sous-utilise la machine, trop de partitions noie le calcul sous
le coût de planification.

Le point à retenir est le coût du ***shuffle*** : une opération comme `groupBy` ou `join`
doit regrouper les lignes de même clé, donc **redistribuer les données entre partitions**.
C'est l'opération la plus coûteuse de Spark, car elle écrit sur disque et passe par le
réseau. L'essentiel de l'optimisation consiste à en limiter le nombre.
""" ),

code("""
print(f"Partitions issues de la lecture : {courses.rdd.getNumPartitions()}")

agregat = courses.groupBy("PULocationID").count()
print(f"Apres groupBy (shuffle)         : {agregat.rdd.getNumPartitions()}"
      f"  <- valeur de spark.sql.shuffle.partitions")

# repartition redistribue avec shuffle, coalesce regroupe sans shuffle
print(f"Apres coalesce(4), sans shuffle : {courses.coalesce(4).rdd.getNumPartitions()}")

lignes_par_partition = (courses.rdd.mapPartitions(lambda p: [sum(1 for _ in p)]).collect())
print(f"\\nRepartition des lignes : {lignes_par_partition}")
print(f"Desequilibre max/min : "
      f"x{max(lignes_par_partition) / max(min(lignes_par_partition), 1):.2f}")
"""),

md("""
### Lecture du résultat : un déséquilibre total

La répartition obtenue est `[0, 0, 0, 10905067, 0, 0, 0, 0]` : **les 10,9 millions de
lignes sont dans une seule partition sur huit**. Sept cœurs sur huit restent inactifs.

La cause est dans le fichier source. Spark découpe un Parquet selon ses *row groups*, pas
selon les blocs HDFS. Ce fichier n'en contient qu'un seul, donc une seule tâche peut le
lire. Le découpage HDFS en 2 blocs ne change rien : il concerne le stockage, pas le
parallélisme de lecture.

**C'est la leçon la plus concrète de cette étape.** Un fichier mal découpé annule le
bénéfice du cluster, quelle que soit sa taille. Deux remèdes :

- `repartition(n)` après la lecture, qui force la redistribution mais coûte un *shuffle* ;
- **réécrire les données en plusieurs fichiers**, ce que fait précisément la section
  suivante.

On observe aussi que le `groupBy` produit **1 partition et non 16**. Ce n'est pas une
erreur : l'exécution adaptative, activée dans la SparkSession, constate que le résultat
tient en une partition et fusionne les 16 prévues. Spark ajuste son plan en cours
d'exécution, d'après les volumes réellement observés.

## 2.7 Écriture en Parquet partitionné

Nous écrivons une copie partitionnée par jour. Chaque valeur de la clé devient un
sous-répertoire, ce qui permet au moteur d'ignorer entièrement les répertoires non concernés
par un filtre : c'est l'élagage de partitions (*partition pruning*).
"""),

code("""
CHEMIN_PARTITIONNE = "hdfs://localhost:9000/nyc_taxi/parquet_partitionne"

preparees = (courses
             .withColumn("jour", F.dayofmonth("tpep_pickup_datetime"))
             .filter(F.col("tpep_pickup_datetime").between("2016-01-01", "2016-02-01")))

debut = time.perf_counter()
(preparees.write.mode("overwrite").partitionBy("jour")
 .parquet(CHEMIN_PARTITIONNE))
print(f"Ecriture partitionnee : {time.perf_counter() - debut:.1f} s")
print(hdfs(f"hdfs dfs -ls {CHEMIN_PARTITIONNE}").split(chr(10))[1][:90] + " ...")
"""),

code("""
# Interet mesure : lire un seul jour depuis le jeu partitionne contre le fichier complet
debut = time.perf_counter()
n_part = spark.read.parquet(CHEMIN_PARTITIONNE).filter(F.col("jour") == 15).count()
t_part = time.perf_counter() - debut

debut = time.perf_counter()
n_plein = courses.filter(F.dayofmonth("tpep_pickup_datetime") == 15).count()
t_plein = time.perf_counter() - debut

print(f"Jeu partitionne : {n_part:,} lignes en {t_part:.2f} s".replace(",", " "))
print(f"Fichier complet : {n_plein:,} lignes en {t_plein:.2f} s".replace(",", " "))
print(f"Gain : x{t_plein / t_part:.1f}")
print("\\nL'elagage de partitions evite d'ouvrir les 30 autres repertoires.")
"""),
    ]

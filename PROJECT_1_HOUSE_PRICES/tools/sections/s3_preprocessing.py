"""Section 3 du pipeline : PRETRAITEMENT."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-3"></a>
# Étape 3 : PRÉTRAITEMENT

**Objectif** : un jeu propre et sans valeur manquante, sans qu'aucune information de la
validation ne contamine l'entraînement.

**L'ordre des opérations est la partie critique :**

```
1. DÉCOUPER train / validation        <- avant toute modification
2. Apprendre les règles de nettoyage  <- sur le TRAIN uniquement
3. Appliquer ces règles                <- aux DEUX jeux, à l'identique
```

Exemple : imputer `LotFrontage` par la médiane du quartier. Si on nettoie **puis** découpe,
la médiane est calculée sur les 1 460 maisons, dont celles qui finiront en validation : le
modèle a vu indirectement le jeu de test et le score devient optimiste. L'erreur est
subtile, elle ne provoque aucun message et produit simplement de beaux chiffres faux.

> **Règle générale** : toute statistique calculée à partir des données (médiane, mode,
> catégories, mise à l'échelle) doit être estimée sur le train seul. C'est le contrat
> `fit` / `transform` de scikit-learn, que nous suivons pour pouvoir réutiliser les mêmes
> objets au déploiement.
"""),

code("""
# Imports de l'etape 3
import joblib
from sklearn.model_selection import train_test_split

brut = pd.read_csv(DATA_RAW / "train.csv", keep_default_na=False, na_values=["NA", ""])
print(f"Donnees brutes : {brut.shape[0]} lignes x {brut.shape[1]} colonnes")
"""),

md("""
## 3.1 Découpage entraînement / validation

| Paramètre | Valeur | Justification |
|---|---|---|
| Validation | 20 %, 292 maisons | Moins rendrait les métriques instables, plus priverait l'entraînement |
| Graine | 42, fixée | Sans elle, chaque exécution donnerait des chiffres différents |
| Stratification | sur `OverallQual` | Voir ci-dessous |

**Pourquoi stratifier** : un tirage purement aléatoire peut placer la quasi-totalité des
maisons haut de gamme d'un seul côté. Sur 18 maisons notées 10, le risque est réel. On ne
peut pas stratifier sur une variable continue, mais `OverallQual` est le meilleur substitut
($r = 0{,}82$ avec le prix).
"""),

code("""
# Decoupage stratifie 80 / 20
# Les notes extremes (1, 2, 10) comptent peu d'individus : on les regroupe
# pour permettre la stratification, sans modifier les donnees elles-memes.
strate = brut.OverallQual.clip(lower=3, upper=9)

train_brut, valid_brut = train_test_split(
    brut, test_size=0.20, random_state=GRAINE, stratify=strate)

print(f"Entrainement : {len(train_brut):4d} maisons ({len(train_brut)/len(brut):.1%})")
print(f"Validation   : {len(valid_brut):4d} maisons ({len(valid_brut)/len(brut):.1%})")
assert not (set(train_brut.Id) & set(valid_brut.Id)), "Chevauchement entre les deux jeux"
print("Controle : aucun chevauchement.")
"""),

code("""
# Le decoupage est-il equilibre ?
# Test de Kolmogorov-Smirnov : H0 = meme distribution. p > 0.05 est le resultat voulu.
comparaison = pd.DataFrame({
    "Entrainement": train_brut.SalePrice.describe(),
    "Validation": valid_brut.SalePrice.describe(),
})
display(comparaison.round(0))

ks = stats.ks_2samp(train_brut.SalePrice, valid_brut.SalePrice)
print(f"Kolmogorov-Smirnov : statistique = {ks.statistic:.4f}, p-value = {ks.pvalue:.3f}")
print("p > 0.05 : les deux jeux suivent la meme distribution de prix."
      if ks.pvalue > 0.05 else "p < 0.05 : les distributions different.")
"""),

code("""
# Visualisation du decoupage
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

bornes = np.linspace(brut.SalePrice.min(), brut.SalePrice.max(), 45)
axes[0].hist(train_brut.SalePrice, bins=bornes, color=viz.BLEU, alpha=0.85, density=True,
             label=f"entrainement (n={len(train_brut)})")
axes[0].hist(valid_brut.SalePrice, bins=bornes, color=viz.ORANGE, alpha=0.65, density=True,
             label=f"validation (n={len(valid_brut)})")
axes[0].set_xlabel("Prix de vente")
axes[0].set_ylabel("Densite")
axes[0].legend()
viz.format_dollars(axes[0], "x")
viz.titrer(axes[0], "Distribution des prix dans les deux jeux",
           f"Test KS : p = {ks.pvalue:.3f}, distributions compatibles")

part_tr = train_brut.OverallQual.value_counts(normalize=True).sort_index()
part_va = valid_brut.OverallQual.value_counts(normalize=True).sort_index()
qualites = sorted(set(part_tr.index) | set(part_va.index))
x = np.arange(len(qualites))
axes[1].bar(x - 0.2, [100 * part_tr.get(q, 0) for q in qualites], width=0.4,
            color=viz.BLEU, label="entrainement")
axes[1].bar(x + 0.2, [100 * part_va.get(q, 0) for q in qualites], width=0.4,
            color=viz.ORANGE, label="validation")
axes[1].set_xticks(x, qualites)
axes[1].set_xlabel("OverallQual")
axes[1].set_ylabel("Part des maisons (%)")
axes[1].legend()
viz.titrer(axes[1], "Verification de la stratification",
           "Les proportions de chaque niveau sont preservees")

fig.tight_layout()
viz.sauver(fig, "fig_10_decoupage")
plt.show()
"""),

md("""
**1 168 maisons d'entraînement, 292 de validation.** Distributions compatibles
(Kolmogorov-Smirnov, $p = 0{,}85$), stratification respectée. Condition nécessaire à une
évaluation honnête.

> À partir d'ici, le jeu de validation est mis de côté jusqu'à l'étape 6.
"""),

md("""
## 3.2 Anomalies (jeu d'entraînement seulement)

Sur l'entraînement, écarter les 2 ventes du § 2.7 est légitime : on choisit ce sur quoi le
modèle apprend. Sur la validation, ce serait de la triche : ce jeu simule les données
futures, et en production on ne choisit pas les biens à estimer.
"""),

code("""
# Suppression des 2 ventes non representatives (cf. section 2.7)
ANOMALIES = [524, 1299]   # tres grandes maisons vendues en condition "Partial"

print("Repartition des 2 anomalies apres decoupage :")
print(f"  dans l'entrainement : {train_brut.Id.isin(ANOMALIES).sum()}")
print(f"  dans la validation  : {valid_brut.Id.isin(ANOMALIES).sum()}")

avant = len(train_brut)
train_brut = train_brut[~train_brut.Id.isin(ANOMALIES)].copy()
print(f"\\nEntrainement : {avant} -> {len(train_brut)} lignes")
print(f"Validation   : {len(valid_brut)} lignes (inchangee, par principe)")
"""),

code("""
# Controles d'integrite
print(f"Doublons : {train_brut.duplicated().sum()} lignes identiques, "
      f"{train_brut.drop(columns=['Id']).duplicated().sum()} hors identifiant\\n")

regles = {
    "YearRemodAdd >= YearBuilt": (train_brut.YearRemodAdd >= train_brut.YearBuilt).all(),
    "YrSold >= YearBuilt": (train_brut.YrSold >= train_brut.YearBuilt).all(),
    "SalePrice > 0": (train_brut.SalePrice > 0).all(),
    "GrLivArea = somme des etages": (
        train_brut.GrLivArea
        == train_brut["1stFlrSF"] + train_brut["2ndFlrSF"] + train_brut.LowQualFinSF).all(),
}
for regle, ok in regles.items():
    print(f"  [{'OK' if ok else 'ECHEC'}] {regle}")
assert all(regles.values())
"""),

md("""
**Le tirage a envoyé une anomalie dans chaque jeu.** L'une était dans l'entraînement et a
été supprimée, l'autre est tombée dans la validation et **y reste**. L'entraînement passe
de 1 168 à **1 167 lignes**.

C'est involontaire, mais c'est la meilleure illustration de la règle. Le modèle sera donc
évalué, entre autres, sur une maison de plus de 4 000 pi² vendue à moins de 40 \\$/pi²,
qu'il va nécessairement mal prédire. **C'est voulu** : une métrique calculée sur un jeu
expurgé de ses cas difficiles mesurerait la qualité du nettoyage, pas celle du modèle. Nous
verrons à l'étape 6 que cette maison pèse lourd, et nous le dirons.

Par ailleurs : aucun doublon, aucune incohérence logique.
"""),

md("""
## 3.3 Valeurs manquantes

| Groupe | Colonnes | Remplissage | Justification |
|---|---|---|---|
| A, absence catégorielle | 14 (`PoolQC`, `Alley`, `Bsmt*`, `Garage*`) | `"Absent"` | Concordance 100 % prouvée au § 2.6 |
| B, absence numérique | `MasVnrArea`, `GarageYrBlt` | `0` / `YearBuilt` | Surface nulle si pas d'équipement |
| C, contextuelle | `LotFrontage` | Médiane du quartier, apprise sur le train | Écart-type inter-quartiers de 20,6 pieds |
| D, simple | `Electrical`, `MasVnrType` | Mode du train | 1 et 8 lignes |

La logique est encapsulée dans une classe `fit` / `transform` : pas de fuite par
construction, traitement identique du train, de la validation et des saisies utilisateur,
et réutilisation directe à l'étape 7 sans réécrire le code.
"""),

code("""
# Les classes du pipeline sont definies dans tools/pipeline_maisons.py, pas dans
# le notebook. Raison technique : joblib serialise les objets par REFERENCE a
# leur module. Une classe definie ici serait enregistree sous le nom
# `__main__.NettoyeurMaisons`, introuvable depuis l'application Streamlit de l'etape 7.
# Leur code source est affiche ci-dessous, rien n'est cache.
import inspect
from IPython.display import Code
from pipeline_maisons import NettoyeurMaisons

Code(inspect.getsource(NettoyeurMaisons), language="python")
"""),

code("""
# fit sur le TRAIN, transform sur les DEUX jeux
nettoyeur = NettoyeurMaisons()
train = nettoyeur.fit_transform(train_brut)   # apprend ET applique
valid = nettoyeur.transform(valid_brut)       # applique seulement

print(f"Entrainement : {train.shape}  |  Validation : {valid.shape}")
print(f"Valeurs manquantes restantes : {train.isna().sum().sum()} (train), "
      f"{valid.isna().sum().sum()} (valid)")
assert train.isna().sum().sum() == 0 and valid.isna().sum().sum() == 0
"""),

code("""
# CONTROLE ANTI-FUITE
# Principe : on refait un fit sur le TRAIN seul, puis sur train + valid.
# Si les statistiques apprises different, c'est que le jeu de validation
# influencerait le nettoyage, donc qu'il y aurait fuite.
temoin = NettoyeurMaisons().fit(train_brut)
contamine = NettoyeurMaisons().fit(pd.concat([train_brut, valid_brut]))

ecarts = (temoin.lotfrontage_quartier_ - contamine.lotfrontage_quartier_).abs()
print(f"Medianes de LotFrontage par quartier, train seul vs train + validation :")
print(f"  {(ecarts > 1e-9).sum()} quartiers sur {len(ecarts)} donnent une valeur differente")
print(f"  ecart maximal : {ecarts.max():.1f} pieds")
print(f"  mediane globale : {temoin.lotfrontage_globale_:.1f} (train) "
      f"vs {contamine.lotfrontage_globale_:.1f} (train+valid)")
print("\\nCes valeurs different : c'est exactement l'information qui aurait fuite")
print("si nous avions nettoye avant de decouper.")
"""),

md("""
Zéro valeur manquante, et la vérification anti-fuite est explicite : les médianes apprises
sur le train seul **diffèrent** de celles apprises sur l'ensemble, pour 11 quartiers sur 25,
avec un écart maximal de 30 pieds. **C'est exactement la quantité d'information qui aurait
fuité.** La fuite n'est pas un risque théorique, elle est mesurable.

**Un choix à expliciter : `GarageYrBlt`.** Pour les 81 maisons sans garage, remplir par 0
créerait une année à 2 000 unités du reste de la distribution, ce qui fabrique un découpage
artificiel pour un arbre et est catastrophique pour une régression. Nous remplissons par
`YearBuilt` : la valeur reste plausible, et l'information "pas de garage" est déjà portée
par `GarageArea = 0`. Bon exemple de cas où l'option la plus simple n'est pas la meilleure.
"""),

md("""
## 3.4 Variables écartées

Supprimer une variable est une perte définitive. Nous n'écartons que ce dont l'inutilité est
démontrable, et refusons les suppressions fondées sur une corrélation faible, qui ne dit
rien des relations non linéaires.
"""),

code("""
# Justification chiffree des deux suppressions
print("1) 'Id' : identifiant administratif")
print(f"   {brut.Id.nunique()} valeurs distinctes pour {len(brut)} lignes.")
print(f"   Correlation avec le prix : r = {brut.Id.corr(brut.SalePrice):+.4f} (bruit pur).\\n")

print("2) 'Utilities' : type de raccordement aux reseaux")
vc = brut.Utilities.value_counts()
print(f"   Repartition : {vc.to_dict()}")
print(f"   Modalite dominante : {100 * vc.iloc[0] / len(brut):.2f} %.")
print("   Une variable quasi constante ne separe aucun groupe.\\n")

print("Autres variables dominees a plus de 99 % par une modalite :")
for col in brut.drop(columns=["Id", "SalePrice"]).columns:
    part = 100 * brut[col].value_counts(dropna=False).iloc[0] / len(brut)
    if part >= 99.0:
        statut = "RETIREE" if col in NettoyeurMaisons.A_RETIRER else "conservee"
        print(f"   {col:14s} {part:6.2f} %   -> {statut}")
"""),

md("""
Trois variables dépassent 99 % de concentration mais sont **conservées** : `Street`,
`PoolArea`, `PoolQC`. La différence avec `Utilities` tient à la **nature de la minorité** :
`Utilities` n'a qu'une seule maison différente, dont un arbre ne peut tirer aucune règle
généralisable. `PoolQC` a 7 maisons avec piscine, peu nombreuses mais nettement plus chères,
et l'information est réelle.

La règle n'est donc pas "supprimer au-dessus de 99 %" mais **"supprimer quand la minorité ne
peut pas porter de signal"**. Résultat : 2 colonnes retirées sur 81.
"""),

md("""
## 3.5 Vérification et sauvegarde
"""),

code("""
# Batterie de controles avant de figer les donnees
controles = {
    "Aucune valeur manquante (train)": train.isna().sum().sum() == 0,
    "Aucune valeur manquante (valid)": valid.isna().sum().sum() == 0,
    "Memes colonnes dans les deux jeux": list(train.columns) == list(valid.columns),
    "'Id' retire": "Id" not in train.columns,
    "'Utilities' retiree": "Utilities" not in train.columns,
    "Aucun prix nul ou negatif": (train.SalePrice > 0).all() and (valid.SalePrice > 0).all(),
    "Aucune valeur infinie": np.isfinite(
        train.select_dtypes(include=[np.number]).to_numpy()).all(),
}
for libelle, ok in controles.items():
    print(f"  [{'OK   ' if ok else 'ECHEC'}] {libelle}")
assert all(controles.values())

cat_cols = train.select_dtypes(exclude=[np.number]).columns
inconnues = {c: sorted(set(valid[c]) - set(train[c])) for c in cat_cols
             if set(valid[c]) - set(train[c])}
print(f"\\nModalites presentes en validation mais absentes du train : {len(inconnues)}")
for c, mods in inconnues.items():
    print(f"   {c}: {mods}")
print("-> gerees a l'etape 4 par l'option handle_unknown de l'encodeur.")
"""),

code("""
# Sauvegarde des artefacts
train.to_csv(DATA_PROC / "train_clean.csv", index=False)
valid.to_csv(DATA_PROC / "valid_clean.csv", index=False)
joblib.dump(nettoyeur, MODELS / "nettoyeur.joblib")

for chemin in [DATA_PROC / "train_clean.csv", DATA_PROC / "valid_clean.csv",
               MODELS / "nettoyeur.joblib"]:
    print(f"  {chemin.relative_to(PROJET)!s:40s} {chemin.stat().st_size / 1024:7.1f} Ko")

# L'objet sauvegarde doit etre reutilisable tel quel a l'etape 7
recharge = joblib.load(MODELS / "nettoyeur.joblib")
assert recharge.transform(valid_brut).equals(valid), "Le nettoyeur recharge differe"
print("\\nControle : le nettoyeur recharge produit un resultat identique.")
"""),

md("""
## 3.6 Garanties obtenues

1. Zéro valeur manquante, vérifié par assertion.
2. Zéro fuite, démontré par le test comparatif des médianes.
3. Le nettoyeur rechargé depuis le disque produit un résultat identique.
4. Jeu de validation intact : ni nettoyé, ni utilisé pour apprendre.

**Limites** : l'imputation par la médiane réduit artificiellement la variance de
`LotFrontage`. Écarter 2 observations reste un jugement, argumenté et isolé dans une
constante `ANOMALIES` facile à modifier pour tester son effet.
"""),
    ]

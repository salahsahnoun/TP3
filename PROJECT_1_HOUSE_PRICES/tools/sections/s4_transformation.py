"""Section 4 du pipeline : TRANSFORMATION et FEATURE ENGINEERING."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-4"></a>
# Étape 4 : TRANSFORMATION

**Objectif** : créer des variables, encoder le texte, mettre à l'échelle là où c'est
nécessaire.

**Décision structurante : deux pipelines, pas un.** Les modèles n'ont pas les mêmes besoins.

| Besoin | Arbres (TFDF, RF, GB) | Régression linéaire |
|---|---|---|
| Encodage des catégorielles | Non, natif | Obligatoire |
| Mise à l'échelle | Non | Oui |
| Correction de l'asymétrie | Non, seul l'ordre compte | Oui |
| Multicolinéarité | Gérée par construction | Régularisation |

Appliquer le traitement linéaire à TFDF détruirait son principal atout sans bénéfice.

> **Pipeline A (arbres)** : variables créées, catégorielles en texte.
> **Pipeline B (linéaire)** : variables créées, encodage, log, normalisation.
>
> Les deux partent du même découpage et des mêmes variables créées : seul diffère le
> traitement imposé par chaque famille. C'est ce qui rend la comparaison équitable.

**Risque de fuite** : le feature engineering du § 4.1 est ligne à ligne, donc
structurellement incapable de fuiter. L'encodage et la normalisation du § 4.3 utilisent des
statistiques globales : ils passent par un `fit` sur le train seul, vérifié au § 4.4.
"""),

code("""
# Imports de l'etape 4
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

train = pd.read_csv(DATA_PROC / "train_clean.csv")
valid = pd.read_csv(DATA_PROC / "valid_clean.csv")
print(f"Entrainement : {train.shape}  |  Validation : {valid.shape}")
"""),

md("""
## 4.1 Feature engineering

Chacune des 13 variables répond à un constat chiffré de l'étape 2.

| Variable créée | Constat qui la justifie |
|---|---|
| `SurfaceTotale` | 5 variables de surface redondantes (§ 2.3) |
| `TotalSdB` | 4 variables de salle de bain dispersées |
| `SurfaceExterieure` | 5 variables de terrasse à inflation de zéros (§ 2.6) |
| `AgeVente`, `AgeRenovation` | `YearBuilt` a une relation non linéaire (§ 2.4c) |
| `EstRenovee`, `EstNeuve` | Information cachée dans deux dates |
| 5 indicateurs de présence | Séparer "existe ?" de "combien ?" (§ 2.6) |
| `QualiteFoisSurface` | Interaction entre les deux variables dominantes (§ 2.2) |

**Pourquoi `QualiteFoisSurface`** : une régression linéaire est structurellement incapable
de modéliser une interaction. Or 100 pi² de plus ne valent pas le même prix dans une maison
de qualité 3 et de qualité 9.
"""),

code("""
# Les classes du pipeline sont definies dans tools/pipeline_maisons.py, pas dans
# le notebook. Raison technique : joblib serialise les objets par REFERENCE a
# leur module. Une classe definie ici serait enregistree sous le nom
# `__main__.ConstructeurVariables`, introuvable depuis l'application Streamlit de l'etape 7.
# Leur code source est affiche ci-dessous, rien n'est cache.
import inspect
from IPython.display import Code
from pipeline_maisons import ConstructeurVariables

Code(inspect.getsource(ConstructeurVariables), language="python")
"""),

code("""
# Application aux deux jeux
constructeur = ConstructeurVariables()
train_fe = constructeur.fit_transform(train)
valid_fe = constructeur.transform(valid)

nouvelles = ConstructeurVariables.nouvelles()
print(f"{len(nouvelles)} variables creees : {train.shape[1]} -> {train_fe.shape[1]} colonnes")
display(train_fe[nouvelles].describe().T.round(2))
"""),

code("""
# Les variables creees apportent-elles quelque chose ?
# Critere : leur correlation depasse-t-elle celle des variables d'origine ?
y_train_log = np.log1p(train_fe.SalePrice)

lignes = []
for creee, sources in [
        ("SurfaceTotale", ["GrLivArea", "TotalBsmtSF", "1stFlrSF"]),
        ("TotalSdB", ["FullBath", "HalfBath", "BsmtFullBath"]),
        ("AgeVente", ["YearBuilt", "YrSold"]),
        ("QualiteFoisSurface", ["OverallQual", "GrLivArea"]),
        ("SurfaceExterieure", ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch"])]:
    r_creee = abs(train_fe[creee].corr(y_train_log))
    r_sources = {s: abs(train_fe[s].corr(y_train_log)) for s in sources}
    meilleure = max(r_sources, key=r_sources.get)
    lignes.append({"variable_creee": creee, "r_creee": round(r_creee, 3),
                   "meilleure_source": meilleure,
                   "r_source": round(r_sources[meilleure], 3),
                   "gain": round(r_creee - r_sources[meilleure], 3)})
display(pd.DataFrame(lignes).sort_values("gain", ascending=False))

print("Indicateurs de presence, ecart de prix median :")
for ind in ["ASousSol", "AGarage", "ACheminee", "APiscine", "AEtage"]:
    med = train_fe.groupby(ind).SalePrice.median()
    if len(med) == 2:
        print(f"  {ind:11s} sans {med[0]:>9,.0f} $ | avec {med[1]:>9,.0f} $ "
              f"| x{med[1]/med[0]:.2f}  (n_avec = {int(train_fe[ind].sum())})")
"""),

code("""
# Trois variables creees, chacune avec sa propre echelle
A_TRACER = [("SurfaceTotale", "GrLivArea", "pi2"),
            ("QualiteFoisSurface", "OverallQual", "note x pi2 / 1000"),
            ("AgeVente", "YearBuilt", "annees")]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, (creee, source, unite) in zip(axes, A_TRACER):
    ax.scatter(train_fe[creee], y_train_log, s=11, color=viz.BLEU, alpha=0.45,
               edgecolors="none")
    # Tendance locale : mediane du log-prix par decile
    deciles = pd.qcut(train_fe[creee], 10, duplicates="drop")
    tendance = y_train_log.groupby(deciles, observed=False).median()
    ax.plot([iv.mid for iv in tendance.index], tendance.values,
            color=viz.ORANGE, lw=2.2, marker="o", ms=4)
    r_c = abs(train_fe[creee].corr(y_train_log))
    r_s = abs(train_fe[source].corr(y_train_log))
    ax.set_xlabel(f"{creee} ({unite})")
    ax.set_ylabel("log(prix)")
    viz.titrer(ax, creee, f"|r| = {r_c:.3f}  (contre {r_s:.3f} pour {source})")

fig.tight_layout()
viz.sauver(fig, "fig_11_variables_creees")
plt.show()
"""),

md("""
**Les agrégats tiennent leur promesse** : `SurfaceTotale` (0,829 contre 0,727 pour
`GrLivArea`) et `QualiteFoisSurface` (0,839 contre 0,818) battent leurs sources.
`QualiteFoisSurface` devient la variable la plus prédictive du jeu.

**`AgeVente` a un gain nul**, sa corrélation étant l'exacte opposée de `YearBuilt`. Elle
reste utile : elle est interprétable, et surtout elle **découple l'âge de la date absolue**.
Un modèle entraîné sur `YearBuilt` apprend que "1950 = pas cher" ; avec `AgeVente` il
apprend que "60 ans = pas cher", règle valable pour des ventes futures.

**Les indicateurs de présence sont discriminants**, sauf `APiscine` avec 5 maisons
concernées : l'écart n'est pas fiable, nous le gardons sans en attendre grand-chose.

**Limite assumée** : `SurfaceTotale` compte le sous-sol sans pondération, alors qu'il vaut
commercialement moins. Pondérer introduirait un paramètre arbitraire à calibrer, donc un
risque de surapprentissage. Les surfaces d'origine restant disponibles, les modèles peuvent
corriger eux-mêmes.
"""),

md("""
## 4.2 Encodage des variables ordinales

Pour `ExterQual` (`Fa < TA < Gd < Ex`) :

| Encodage | Problème |
|---|---|
| One-hot, 4 colonnes | Perd l'ordre |
| Alphabétique (`Ex`=0, `Fa`=1...) | Ordre faux, place `Ex` au plus bas |
| **Ordinal métier** (`Fa`=2 ... `Ex`=5) | Correct |

Bénéfices : 1 colonne au lieu de 4, un seul test `>= 4` sépare `{Gd, Ex}` de `{Fa, TA}`, et
le coefficient se lit comme la valeur d'un cran de qualité.

Nous ne décrétons pas l'ordre, nous **vérifions** que le prix médian croît avec l'échelle.
"""),

code('''
# Echelles ordinales, d'apres la documentation du dataset
# Echelle commune a 10 colonnes de qualite / etat :
#   Absent < Po(or) < Fa(ir) < TA (Typical) < Gd (Good) < Ex(cellent)
QUALITE = {"Absent": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ECHELLES_ORDINALES = {
    "ExterQual": QUALITE, "ExterCond": QUALITE,
    "BsmtQual": QUALITE, "BsmtCond": QUALITE,
    "HeatingQC": QUALITE, "KitchenQual": QUALITE,
    "FireplaceQu": QUALITE, "GarageQual": QUALITE, "GarageCond": QUALITE,
    "PoolQC": QUALITE,
    "BsmtExposure": {"Absent": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"Absent": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"Absent": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"Absent": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Fence": {"Absent": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4,
                   "Min2": 5, "Min1": 6, "Typ": 7},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
}
print(f"{len(ECHELLES_ORDINALES)} variables ordinales a encoder")
'''),

code("""
# Le prix median croit-il avec l'echelle supposee ?
# Spearman mesure la monotonie, pas la linearite.
controles = []
for col, echelle in ECHELLES_ORDINALES.items():
    med = (train_fe.groupby(col).SalePrice.median()
           .reindex([m for m in echelle if m in train_fe[col].unique()]))
    rangs = [echelle[m] for m in med.index]
    rho = stats.spearmanr(rangs, med.values).statistic if len(med) > 2 else np.nan
    controles.append({
        "variable": col, "modalites": len(med),
        "prix_min": f"{med.min():,.0f}", "prix_max": f"{med.max():,.0f}",
        "rho_Spearman": round(rho, 3) if not np.isnan(rho) else None,
        "monotone": "oui" if (np.isnan(rho) or rho > 0.75) else "NON",
    })
tableau = pd.DataFrame(controles).sort_values("rho_Spearman", na_position="last")
display(tableau)
print(f"Echelles confirmees : {(tableau.monotone == 'oui').sum()} / {len(tableau)}")
"""),

md("""
**13 échelles sur 20 sont confirmées** ($\\rho > 0{,}75$). Les variables de qualité
atteignent $\\rho = 1$, soit une monotonie parfaite.

**7 ne le sont pas**, pour trois raisons :

- **Relation inversée mais monotone** : `LotShape` ($\\rho = -1$), les terrains irréguliers
  se vendent *plus cher* (215 000 contre 147 000 \\$), car ils sont typiques des
  lotissements récents en impasse. Idem `LandSlope` et `Fence`. Une relation décroissante
  reste parfaitement exploitable.
- **L'état n'est pas la qualité** : `ExterCond` ($\\rho = 0{,}70$), même phénomène que
  `OverallCond` au § 2.2.
- **Effectifs trop faibles** : `BsmtFinType1`/`2` et `Functional`.

> **Décision** : on garde l'encodage ordinal pour les 20. Une relation décroissante ou non
> monotone n'invalide pas l'encodage, elle indique seulement que le modèle devra travailler
> un peu plus.
"""),

md("""
## 4.3 Construction des deux pipelines

**Pipeline A** : variables créées, encodage ordinal, nominales en texte.

**Pipeline B** : `log1p` puis normalisation sur les numériques asymétriques, normalisation
sur les autres, binaires inchangées, one-hot sur les nominales.

**`handle_unknown="ignore"` est indispensable** : l'étape 3 a montré que la validation
contient 4 modalités absentes du train. Sans cette option l'encodeur lèverait une exception.
C'est exactement le problème que rencontrera l'application de l'étape 7 face à une saisie
inattendue.
"""),

code("""
import inspect
from IPython.display import Code
from pipeline_maisons import PreparationArbres

Code(inspect.getsource(PreparationArbres), language="python")
"""),

code("""
prep_arbres = PreparationArbres(ECHELLES_ORDINALES)
X_train_arbres = prep_arbres.fit_transform(train)
X_valid_arbres = prep_arbres.transform(valid)

y_train = train.SalePrice.copy()
y_valid = valid.SalePrice.copy()
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

n_texte = X_train_arbres.select_dtypes(exclude=[np.number]).shape[1]
print(f"PIPELINE A : {X_train_arbres.shape} / {X_valid_arbres.shape}")
print(f"  dont {n_texte} colonnes categorielles en texte (traitees nativement par TFDF)")
"""),

code('''
# PIPELINE B : modeles lineaires
base = X_train_arbres
colonnes_nominales = base.select_dtypes(exclude=[np.number]).columns.tolist()
colonnes_numeriques = base.select_dtypes(include=[np.number]).columns.tolist()

# Les binaires creees sont deja en 0/1 : les normaliser n'apporte rien
BINAIRES = ["EstRenovee", "EstNeuve", "ASousSol", "AGarage",
            "ACheminee", "APiscine", "AEtage"]
colonnes_numeriques = [c for c in colonnes_numeriques if c not in BINAIRES]

# Separation selon l'asymetrie, mesuree sur le TRAIN uniquement
SEUIL_ASYMETRIE = 0.75
asym_train = base[colonnes_numeriques].skew()
col_asymetriques = asym_train[asym_train.abs() > SEUIL_ASYMETRIE].index.tolist()
col_normales = [c for c in colonnes_numeriques if c not in col_asymetriques]

print(f"  asymetriques (|skew| > {SEUIL_ASYMETRIE}) : {len(col_asymetriques)}")
print(f"  quasi symetriques                 : {len(col_normales)}")
print(f"  binaires                          : {len(BINAIRES)}")
print(f"  nominales (one-hot)               : {len(colonnes_nominales)}")

assert (base[col_asymetriques] >= 0).all().all(), "log1p exige des valeurs >= 0"

prep_lineaire = ColumnTransformer([
    ("asymetriques", Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("echelle", StandardScaler())]), col_asymetriques),
    ("normales", StandardScaler(), col_normales),
    ("binaires", "passthrough", BINAIRES),
    ("nominales", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     colonnes_nominales),
], remainder="drop", verbose_feature_names_out=False)

# fit sur le TRAIN uniquement, transform sur les deux
X_train_lin = prep_lineaire.fit_transform(X_train_arbres)
X_valid_lin = prep_lineaire.transform(X_valid_arbres)
noms_lin = prep_lineaire.get_feature_names_out()

print(f"\\nPIPELINE B : {X_train_lin.shape} / {X_valid_lin.shape}")
print(f"  {X_train_arbres.shape[1]} colonnes -> {X_train_lin.shape[1]} apres one-hot")
'''),

md("""
Le one-hot fait passer le jeu de **91 à 246 colonnes**. Le ratio observations/variables
tombe de 13 à **4,7 pour 1**, régime où une régression non régularisée surapprendrait.
D'où Ridge à l'étape 5.

C'est aussi l'avantage pratique de TFDF, chiffres à l'appui : **91 colonnes** contre **246**
pour la même information.
"""),

md("""
## 4.4 Contrôle anti-fuite

Trois tests : cohérence du `fit`, indépendance ligne à ligne, et persistance après
rechargement.
"""),

code("""
# TEST 1 : les parametres viennent-ils du seul jeu d'entrainement ?
moyennes_apprises = prep_lineaire.named_transformers_["normales"].mean_
moyennes_train = X_train_arbres[col_normales].mean().values
moyennes_tout = pd.concat([X_train_arbres, X_valid_arbres])[col_normales].mean().values

ecart_train = np.abs(moyennes_apprises - moyennes_train).max()
ecart_tout = np.abs(moyennes_apprises - moyennes_tout).max()
print(f"Ecart max avec les statistiques du TRAIN seul  : {ecart_train:.10f}")
print(f"Ecart max avec les statistiques de TOUT le jeu : {ecart_tout:.6f}")
assert ecart_train < 1e-9 and ecart_tout > 1e-6
print("\\n[OK] Parametres issus du train seul. L'ecart non nul avec 'tout le jeu'")
print("     confirme que la validation les aurait modifies si elle avait servi.")
"""),

code("""
# TEST 2 : la transformation d'une ligne depend-elle des autres lignes ?
seule = prep_lineaire.transform(prep_arbres.transform(valid.iloc[[0]]))
ecart_max = np.abs(seule - X_valid_lin[[0]]).max()
print(f"Ecart entre transformation isolee et en lot : {ecart_max:.12f}")
assert ecart_max < 1e-9
print("[OK] Chaque ligne est transformee independamment des autres.")
print("     Condition necessaire pour servir une prediction unitaire (etape 7).")
"""),

code("""
# TEST 3 : persistance
joblib.dump({
    "prep_arbres": prep_arbres,
    "prep_lineaire": prep_lineaire,
    "colonnes_arbres": list(X_train_arbres.columns),
    "noms_lineaire": list(noms_lin),
    "echelles_ordinales": ECHELLES_ORDINALES,
}, MODELS / "preprocessor.joblib")

recharge = joblib.load(MODELS / "preprocessor.joblib")
verif = recharge["prep_lineaire"].transform(recharge["prep_arbres"].transform(valid))
assert np.allclose(verif, X_valid_lin)
print(f"[OK] Pipeline sauvegarde "
      f"({(MODELS / 'preprocessor.joblib').stat().st_size / 1024:.1f} Ko) et verifie.")
"""),

code("""
# Sauvegarde des jeux prets pour l'apprentissage
X_train_arbres.to_csv(DATA_PROC / "X_train_arbres.csv", index=False)
X_valid_arbres.to_csv(DATA_PROC / "X_valid_arbres.csv", index=False)
np.save(DATA_PROC / "X_train_lin.npy", X_train_lin)
np.save(DATA_PROC / "X_valid_lin.npy", X_valid_lin)
y_train.to_csv(DATA_PROC / "y_train.csv", index=False)
y_valid.to_csv(DATA_PROC / "y_valid.csv", index=False)
print("Jeux prets pour l'etape 5.")
"""),

md("""
## 4.5 Garanties obtenues

1. Feature engineering ligne à ligne, fuite structurellement impossible.
2. Paramètres d'encodage et de normalisation appris sur le train seul, vérifié
   numériquement.
3. Transformation indépendante du contexte, condition pour servir une prédiction unitaire.
4. Persistance vérifiée.
5. Comparaison équitable préservée.

**Limites** : les 13 variables viennent d'un raisonnement métier, pas d'une recherche
automatique. Le seuil d'asymétrie de 0,75 est une convention. L'encodage ordinal impose des
écarts égaux entre crans, ce qui est discutable ; un target encoding le corrigerait mais
introduirait un risque de fuite difficile à maîtriser sur 1 167 lignes.
"""),
    ]

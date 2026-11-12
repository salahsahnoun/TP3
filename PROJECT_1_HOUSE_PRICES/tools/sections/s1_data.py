"""Section 1 du pipeline : DATA."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-1"></a>
# Étape 1 : DATA

**Objectif** : connaître les données avant de les toucher. Cette étape décrit, elle ne
transforme rien.

**Problématique** : prédire `SalePrice` à partir de 79 caractéristiques. Apprentissage
supervisé, régression.

**Métrique Kaggle** : RMSE sur $\\log(\\text{SalePrice})$. Comme
$\\log(\\hat{y}) - \\log(y) = \\log(\\hat{y}/y)$, elle mesure une erreur **relative**.
Se tromper de 20 % coûte autant sur une maison à 100 000 \\$ que sur une à 500 000. Cela
guidera nos choix à l'étape 5.

**Origine** : dataset Ames Housing (Pr. Dean De Cock), registre officiel des transactions
d'Ames, Iowa, 2006 à 2010. Chaque ligne est une vente réelle.
"""),

code("""
# Imports de l'etape 1 (PROJET, DATA_RAW... viennent de la cellule precedente)
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)

print("Python :", sys.version.split()[0])
print("pandas :", pd.__version__)
print("numpy  :", np.__version__)
"""),

md("""
## 1.1 Chargement

`test.csv` ne contient pas la cible : les vrais prix sont gardés secrets par Kaggle. Toute
évaluation passera donc par un jeu de validation découpé dans `train.csv`.

**Piège à la lecture** : `read_csv` convertit par défaut `"None"` en `NaN`. Or `"None"` est
une modalité légitime de `MasVnrType` (pas de parement en maçonnerie). Une lecture naïve
transforme 864 maisons documentées en données manquantes.
"""),

code("""
# Lecture naive vs lecture correcte
fichiers = sorted(p for p in DATA_RAW.glob("*") if p.is_file())
display(pd.DataFrame([{"fichier": p.name, "taille_Ko": round(p.stat().st_size / 1024, 1)}
                      for p in fichiers]))

CHEMIN_TRAIN = DATA_RAW / "train.csv"
CHEMIN_TEST = DATA_RAW / "test.csv"
A_UN_TEST = CHEMIN_TEST.exists()

naif = pd.read_csv(CHEMIN_TRAIN)

# On desactive la liste par defaut et on declare ce qui est reellement manquant
NA_EXPLICITES = ["NA", ""]
train = pd.read_csv(CHEMIN_TRAIN, keep_default_na=False, na_values=NA_EXPLICITES)
test = (pd.read_csv(CHEMIN_TEST, keep_default_na=False, na_values=NA_EXPLICITES)
        if A_UN_TEST else None)

display(pd.DataFrame({
    "lecture_naive": naif[["MasVnrType", "MasVnrArea"]].isna().sum(),
    "lecture_correcte": train[["MasVnrType", "MasVnrArea"]].isna().sum(),
}))
print("Modalites reelles de MasVnrType :")
print(train.MasVnrType.value_counts(dropna=False).to_string())
"""),

code("""
print(f"train : {train.shape[0]} lignes x {train.shape[1]} colonnes")
if test is not None:
    print(f"test  : {test.shape[0]} lignes x {test.shape[1]} colonnes")
    print(f"Colonne(s) absente(s) de test : {set(train.columns) - set(test.columns)}")
else:
    print("test  : non disponible (necessite l'API Kaggle)")
    print("        sans consequence : evaluation sur un jeu de validation (etape 6)")

display(train[["Id", "MSZoning", "LotArea", "Neighborhood", "OverallQual",
               "YearBuilt", "GrLivArea", "GarageCars", "SaleCondition",
               "SalePrice"]].head())
"""),

md("""
**1 460 observations, 79 variables.** Le ratio de 18 observations par variable est faible :
un modèle trop flexible surapprendra, les réseaux de neurones sont inadaptés, et les
méthodes d'ensemble à base d'arbres sont le bon choix. Cela justifie TFDF.
"""),

md("""
## 1.2 La variable cible
"""),

code("""
# Statistiques descriptives de la cible
y = train["SalePrice"]
display(pd.Series({
    "Observations": f"{y.count():,}",
    "Moyenne": f"{y.mean():,.0f} $",
    "Mediane": f"{y.median():,.0f} $",
    "Ecart-type": f"{y.std():,.0f} $",
    "Minimum": f"{y.min():,.0f} $",
    "Maximum": f"{y.max():,.0f} $",
    "Asymetrie (skewness)": f"{y.skew():.3f}",
    "Aplatissement (kurtosis)": f"{y.kurt():.3f}",
}, name="Valeur").to_frame())
print(f"Rapport max/min : x{y.max() / y.min():.1f}")
"""),

md("""
| Constat | Valeur | Conséquence |
|---|---|---|
| Asymétrie à droite | 1,883 | Une transformation log la rendra quasi symétrique |
| Queues épaisses | kurtosis 6,536 | Les maisons de luxe pèseront lourd |
| Forte amplitude | rapport 21,6 | L'erreur absolue n'a pas le même sens partout |

Moyenne (180 921 \\$) supérieure à la médiane (163 000 \\$) : une minorité de propriétés de
luxe tire la moyenne. La régression linéaire supposant des résidus normaux et de variance
constante, elle marchera mieux sur $\\log(\\text{SalePrice})$. Vérifié à l'étape 2.
"""),

md("""
## 1.3 Typologie des variables

On ne se fie pas aux types détectés par pandas. `MSSubClass` est un entier mais c'est un
code de catégorie. `ExterQual`, `KitchenQual` sont du texte mais **ordinales** : il existe
un ordre naturel qu'un encodage nominal détruirait.
"""),

code("""
# Typologie brute vs typologie corrigee
features = train.drop(columns=["Id", "SalePrice"])

print("Types detectes par pandas :")
for dtype, n in features.dtypes.value_counts().items():
    print(f"  {str(dtype):10s} : {n}")

FAUSSES_NUMERIQUES = ["MSSubClass"]          # codes de categorie
NOTES = ["OverallQual", "OverallCond"]       # notes deja ordinales
ORDINALES = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "GarageFinish", "PoolQC", "Fence", "Functional",
    "LandSlope", "LotShape", "PavedDrive", "CentralAir", "Utilities",
]

num_brut = features.select_dtypes(include=[np.number]).columns.tolist()
cat_brut = features.select_dtypes(exclude=[np.number]).columns.tolist()

numeriques = [c for c in num_brut if c not in FAUSSES_NUMERIQUES + NOTES]
nominales = [c for c in cat_brut if c not in ORDINALES] + FAUSSES_NUMERIQUES
ordinales = ORDINALES + NOTES

print("\\nTypologie corrigee :")
print(f"  Numeriques : {len(numeriques)}")
print(f"  Nominales  : {len(nominales)}")
print(f"  Ordinales  : {len(ordinales)}")
print(f"  TOTAL      : {len(numeriques) + len(nominales) + len(ordinales)}")
"""),

code("""
# Plage, cardinalite et part de zeros des variables numeriques
detail_num = pd.DataFrame({
    "min": features[numeriques].min(),
    "max": features[numeriques].max(),
    "valeurs_distinctes": features[numeriques].nunique(),
    "pct_zeros": (features[numeriques] == 0).mean().mul(100).round(1),
}).sort_values("pct_zeros", ascending=False)
display(detail_num.head(12))
"""),

md("""
`pct_zeros` est la colonne la plus instructive. `PoolArea` (99,5 % de zéros), `3SsnPorch`
(98,4 %), `MiscVal` (96,4 %) ne sont pas des données manquantes : ces zéros signifient
"l'équipement n'existe pas". Ces distributions mélangent deux informations, l'existence et
la taille. Nous les séparerons à l'étape 4.
"""),

code("""
# Cardinalite et concentration des variables categorielles
cat_toutes = nominales + [c for c in ordinales if c not in NOTES]
detail_cat = pd.DataFrame([{
    "variable": col,
    "type": "ordinale" if col in ORDINALES else "nominale",
    "modalites": features[col].nunique(dropna=True),
    "dominante": str(features[col].value_counts(dropna=False).index[0]),
    "pct_dominante": round(100 * features[col].value_counts(dropna=False).iloc[0]
                           / len(features), 1),
} for col in cat_toutes]).sort_values("pct_dominante", ascending=False)
display(detail_cat.head(12))
print(f"Cardinalite maximale : {detail_cat.modalites.max()} "
      f"({detail_cat.loc[detail_cat.modalites.idxmax(), 'variable']})")
"""),

md("""
**`Utilities` est quasi constante** (1 459 maisons sur 1 460) : sans variance, elle ne
discrimine rien, elle sera supprimée à l'étape 3.

**`Neighborhood` compte 25 modalités.** Un encodage one-hot créerait 25 colonnes pour
1 460 lignes. TFDF gère nativement les catégorielles : c'est son principal avantage ici.
"""),

code("""
# Inventaire des valeurs manquantes
nb_nan = train.isna().sum()
nb_nan = nb_nan[nb_nan > 0].sort_values(ascending=False)
display(pd.DataFrame({"nb": nb_nan, "pct": (100 * nb_nan / len(train)).round(2)}))
print(f"Colonnes concernees : {len(nb_nan)} / {train.shape[1]}")
"""),

code("""
# Controles d'integrite
print("Doublons :")
print(f"  Lignes identiques       : {train.duplicated().sum()}")
print(f"  Id dupliques            : {train['Id'].duplicated().sum()}")
print(f"  Doublons hors Id        : {train.drop(columns=['Id']).duplicated().sum()}")

print("\\nCoherence logique :")
for libelle, n in {
    "Renovation avant construction": (train.YearRemodAdd < train.YearBuilt).sum(),
    "Vente avant construction": (train.YrSold < train.YearBuilt).sum(),
    "GrLivArea != somme des etages": (
        train.GrLivArea != train["1stFlrSF"] + train["2ndFlrSF"] + train.LowQualFinSF).sum(),
    "Prix nul ou negatif": (train.SalePrice <= 0).sum(),
    "Garage declare mais surface nulle": (
        (train.GarageCars > 0) & (train.GarageArea == 0)).sum(),
}.items():
    print(f"  [{'OK' if n == 0 else 'A VERIFIER'}] {libelle} : {n}")

print(f"\\nPeriode : ventes {train.YrSold.min()}-{train.YrSold.max()}, "
      f"constructions {train.YearBuilt.min()}-{train.YearBuilt.max()}")
"""),

md("""
19 colonnes ont des valeurs manquantes, mais **15 signifient "équipement absent"**
(`PoolQC`, `Alley`, `Fence`, `Garage*`, `Bsmt*`). Seules 4 demandent une vraie imputation :
`LotFrontage` (259), `MasVnrType` et `MasVnrArea` (8 chacune), `Electrical` (1), soit
**0,23 % des cellules**. Correctement lu, le dataset est remarquablement complet.

Aucun doublon, aucune incohérence. L'identité
`GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF` est vérifiée sur les 1 460 lignes : c'est
une multicolinéarité parfaite, à traiter à l'étape 4.
"""),

md("""
## 1.4 Limites du dataset

À énoncer avant tout résultat, pour ne pas survendre le modèle :

1. **Portée géographique nulle** : uniquement Ames, Iowa, 60 000 habitants. Non
   transposable à un autre marché.
2. **Période atypique** : 2006 à 2010 couvre la bulle immobilière et la crise des subprimes.
3. **Obsolescence** : données vieilles de 15 ans, seules les relations relatives restent
   informatives.
4. **Variables absentes** : ni taux d'intérêt, ni qualité des écoles, ni distance au centre.
5. **Taille modeste** : jeu pédagogique, les intervalles de confiance seront larges.
"""),

code("""
# Sauvegarde de la typologie pour les etapes suivantes
import json

chemin_meta = DATA_PROC / "metadonnees.json"
chemin_meta.write_text(json.dumps({
    "n_lignes": int(len(train)),
    "n_colonnes": int(train.shape[1]),
    "cible": "SalePrice",
    "variables_numeriques": numeriques,
    "variables_nominales": nominales,
    "variables_ordinales": ordinales,
    "colonnes_avec_nan": nb_nan.index.tolist(),
    "a_un_fichier_test": bool(A_UN_TEST),
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Metadonnees : {chemin_meta.relative_to(PROJET)}")
"""),
    ]

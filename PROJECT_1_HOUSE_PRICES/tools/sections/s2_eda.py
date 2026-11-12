"""Section 2 du pipeline : ANALYSE EXPLORATOIRE (EDA)."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-2"></a>
# Étape 2 : EDA

**Objectif** : chaque analyse doit produire une **décision**, pas seulement un graphique.

**Lien avec le notebook original** : il consacre son EDA à la distribution de la cible et
aux histogrammes des variables numériques. Nous les reproduisons (§ 2.1 et § 2.6) et
ajoutons ce qui manque : corrélations, multicolinéarité, nature des valeurs manquantes,
anomalies.

**Règle** : l'EDA se fait sur tout `train.csv` et ne transforme rien. Observer avant de
découper ne crée pas de fuite, tant qu'aucune statistique n'est *apprise* ici pour le
modèle.
"""),

code("""
# Rechargement (lecture correcte, cf. etape 1)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

NA_EXPLICITES = ["NA", ""]
df = pd.read_csv(DATA_RAW / "train.csv", keep_default_na=False, na_values=NA_EXPLICITES)

y = df["SalePrice"]
y_log = np.log1p(y)
features = df.drop(columns=["Id", "SalePrice"])
print(f"Donnees : {df.shape[0]} lignes x {df.shape[1]} colonnes")
"""),

md("""
## 2.1 Distribution de la cible

Décision la plus structurante de l'EDA. Trois angles : histogramme, diagramme
quantile-quantile, test de Shapiro-Wilk.
"""),

code("""
# SalePrice brut vs log(SalePrice)
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

axes[0, 0].hist(y, bins=50, color=viz.BLEU)
axes[0, 0].axvline(y.mean(), color=viz.ORANGE, lw=2, label=f"moyenne {y.mean():,.0f} $")
axes[0, 0].axvline(y.median(), color=viz.ENCRE, lw=2, ls="--",
                   label=f"mediane {y.median():,.0f} $")
axes[0, 0].set_xlabel("Prix de vente")
axes[0, 0].set_ylabel("Nombre de maisons")
axes[0, 0].legend(loc="upper right")
viz.format_dollars(axes[0, 0], "x")
viz.titrer(axes[0, 0], "SalePrice brut", f"asymetrie {y.skew():.2f}, queue droite marquee")

axes[0, 1].hist(y_log, bins=50, color=viz.AQUA)
axes[0, 1].axvline(y_log.mean(), color=viz.ORANGE, lw=2, label="moyenne")
axes[0, 1].axvline(y_log.median(), color=viz.ENCRE, lw=2, ls="--", label="mediane")
axes[0, 1].set_xlabel("log(1 + prix de vente)")
axes[0, 1].set_ylabel("Nombre de maisons")
axes[0, 1].legend(loc="upper right")
viz.titrer(axes[0, 1], "log(SalePrice)", f"asymetrie {y_log.skew():.2f}, quasi symetrique")

for ax, serie, couleur, titre in [
        (axes[1, 0], y, viz.BLEU, "Q-Q plot, brut"),
        (axes[1, 1], y_log, viz.AQUA, "Q-Q plot, log")]:
    (osm, osr), (pente, origine, r) = stats.probplot(serie, dist="norm")
    ax.scatter(osm, osr, s=6, color=couleur, alpha=0.6)
    ax.plot(osm, pente * osm + origine, color=viz.ENCRE, lw=1.5, ls="--")
    ax.set_xlabel("Quantiles theoriques (loi normale)")
    ax.set_ylabel("Quantiles observes")
    ax.grid(True, axis="both")
    viz.titrer(ax, titre, f"R2 = {r**2:.4f}, plus proche de 1 = plus normal")

fig.tight_layout()
viz.sauver(fig, "fig_01_distribution_cible")
plt.show()
"""),

code("""
# Verdict statistique : Shapiro-Wilk (H0 = loi normale)
display(pd.DataFrame({
    "SalePrice brut": [y.skew(), y.kurt(), stats.shapiro(y).pvalue],
    "log(SalePrice)": [y_log.skew(), y_log.kurt(), stats.shapiro(y_log).pvalue],
}, index=["Asymetrie", "Aplatissement", "Shapiro-Wilk (p-value)"]))
"""),

md("""
| Indicateur | Brut | Log |
|---|---|---|
| Asymétrie | 1,883 | **0,121** |
| Aplatissement | 6,536 | **0,810** |
| Shapiro-Wilk | 3,2e-33 | 1,2e-07 |

Le log améliore nettement sans rendre parfait. Shapiro-Wilk rejette toujours la normalité,
mais ce test est très sensible sur de grands échantillons : avec $n = 1460$ il détecte le
moindre écart. Les indicateurs utiles sont l'asymétrie, l'aplatissement et le Q-Q plot.

> **Décision : entraîner sur `log(SalePrice)`**, reconvertir en dollars pour l'évaluation.
>
> 1. La métrique Kaggle est une RMSE sur log : on optimise directement la bonne cible.
> 2. Sur les prix bruts, l'erreur croît avec le prix. Le log stabilise cette variance.
> 3. Le log transforme une erreur absolue en erreur relative, ce qui a un sens économique.
>
> Les métriques de l'étape 6 seront rapportées **en dollars**, après reconversion.
"""),

md("""
## 2.2 Corrélations avec la cible

Pearson mesure la relation **linéaire**. On le calcule contre la cible brute et contre sa
version log : si le log est le bon choix, les corrélations doivent s'améliorer. Limite :
une relation en U donnerait $r \\approx 0$ alors que le lien est fort, d'où l'inspection
visuelle du § 2.4.
"""),

code("""
# Correlations contre la cible brute et contre sa version log
num = df.select_dtypes(include=[np.number]).drop(columns=["Id"])
num_feats = num.drop(columns=["SalePrice"])

corr = pd.DataFrame({"r_brut": num_feats.corrwith(y), "r_log": num_feats.corrwith(y_log)})
corr["gain_log"] = corr.r_log.abs() - corr.r_brut.abs()
corr = corr.sort_values("r_log", ascending=False)

print("10 correlations positives les plus fortes :")
display(corr.head(10).round(3))
print("5 correlations negatives les plus fortes :")
display(corr.tail(5).round(3))
print(f"Correlations ameliorees par le log : {(corr.gain_log > 0).sum()} / {len(corr)}")
"""),

code("""
# Polarite (positif / negatif) donc palette divergente
top = corr.reindex(corr.r_log.abs().sort_values(ascending=False).index).head(18)
top = top.sort_values("r_log")

fig, ax = plt.subplots(figsize=(9, 7))
barres = ax.barh(top.index, top.r_log, height=0.68,
                 color=[viz.BLEU if v > 0 else viz.ROUGE for v in top.r_log])
ax.bar_label(barres, fmt="%.2f", padding=4, fontsize=8.5, color=viz.ENCRE_2)
ax.axvline(0, color=viz.ENCRE_3, lw=1)
ax.set_xlim(-0.35, 0.95)
ax.set_xlabel("Correlation de Pearson avec log(SalePrice)")
ax.grid(True, axis="x"); ax.grid(False, axis="y")
ax.legend(handles=[Patch(color=viz.BLEU, label="relation positive"),
                   Patch(color=viz.ROUGE, label="relation negative")], loc="lower right")
viz.titrer(ax, "Les 18 variables numeriques les plus correlees au prix",
           "La qualite generale domine, devant la surface habitable")
fig.tight_layout()
viz.sauver(fig, "fig_02_correlations")
plt.show()
"""),

md("""
| Rang | Variable | r (log) |
|---|---|---|
| 1 | `OverallQual` | **0,817** |
| 2 | `GrLivArea` | **0,701** |
| 3 | `GarageCars` | 0,681 |
| 4 | `GarageArea` | 0,651 |
| 5 | `TotalBsmtSF` | 0,612 |

**La qualité bat la quantité.** `OverallQual`, simple note de 1 à 10, est plus prédictive
que n'importe quelle surface.

**Le log est validé empiriquement** : `OverallQual` passe de 0,791 à 0,817, `YearBuilt` de
0,523 à 0,587, et 23 variables sur 36 y gagnent.

**Les corrélations négatives sont trompeuses.** `KitchenAbvGr` (-0,148) : 2 cuisines
signale un logement bi-familial, moins valorisé. `OverallCond` (-0,037) : les maisons
anciennes ont une bonne note d'état tout en valant moins cher, la note est relative à
l'âge. La supprimer serait une erreur, les arbres exploitent son interaction avec
`YearBuilt`.
"""),

md("""
## 2.3 Multicolinéarité

Quand $X_1 \\approx X_2$, le modèle linéaire ne peut pas répartir leur effet de façon stable
et les coefficients deviennent ininterprétables. Les arbres y sont insensibles : ils
choisissent simplement l'une des deux à chaque nœud.
"""),

code("""
# Matrice de correlation des variables dominantes
top10 = corr.r_log.abs().sort_values(ascending=False).head(10).index.tolist()
matrice = num[top10 + ["SalePrice"]].corr()

fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(matrice, cmap=viz.CMAP_DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(matrice)), matrice.columns, rotation=45, ha="right")
ax.set_yticks(range(len(matrice)), matrice.columns)
ax.grid(False)
for i in range(len(matrice)):
    for j in range(len(matrice)):
        v = matrice.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.65 else viz.ENCRE)
cb = fig.colorbar(im, ax=ax, shrink=0.75)
cb.set_label("Correlation de Pearson", color=viz.ENCRE_2)
cb.outline.set_visible(False)
viz.titrer(ax, "Correlations entre les 10 variables dominantes",
           "Les blocs fonces hors diagonale signalent la redondance")
fig.tight_layout()
viz.sauver(fig, "fig_03_matrice_correlation")
plt.show()
"""),

code("""
# Paires redondantes (|r| > 0.70)
c = num_feats.corr().abs()
paires = (c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
           .stack().sort_values(ascending=False))
redondantes = paires[paires > 0.70]
display(pd.DataFrame({
    "variable_A": [a for a, _ in redondantes.index],
    "variable_B": [b for _, b in redondantes.index],
    "abs_r": redondantes.values.round(3),
}))
"""),

md("""
| Paire | \\|r\\| | Pourquoi |
|---|---|---|
| `GarageCars` / `GarageArea` | 0,882 | Même information, en voitures ou en pi² |
| `YearBuilt` / `GarageYrBlt` | 0,826 | Garage construit avec la maison |
| `GrLivArea` / `TotRmsAbvGrd` | 0,825 | Plus grand, plus de pièces |
| `TotalBsmtSF` / `1stFlrSF` | 0,820 | Le sous-sol épouse l'emprise au sol |

> **Décisions.** Arbres : on ne supprime rien. Régression linéaire : régularisation
> **Ridge**, qui stabilise les coefficients sans supprimer arbitrairement de variables. Les
> deux : créer des agrégats (`SurfaceTotale`, `TotalSdB`) à l'étape 4.
"""),

md("""
## 2.4 Forme des relations

Un $r$ élevé ne garantit pas la linéarité, et c'est elle qui détermine quels modèles
conviennent.
"""),

code("""
# Style commun aux boites a moustaches de cette section
STYLE_BOITE = dict(
    patch_artist=True,
    medianprops=dict(color=viz.ENCRE, lw=1.6),
    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=viz.ENCRE_3,
                    markeredgecolor="none", alpha=0.5),
    boxprops=dict(edgecolor="none"),
    whiskerprops=dict(color=viz.ENCRE_3),
    capprops=dict(color=viz.ENCRE_3),
)


def colorer(bp, n):
    \"\"\"Rampe sequentielle : la couleur encode le rang de la modalite.\"\"\"
    for i, boite in enumerate(bp["boxes"]):
        boite.set_facecolor(viz.CMAP_SEQ(0.15 + 0.85 * i / max(n - 1, 1)))


fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

qualites = sorted(df.OverallQual.unique())
bp = axes[0, 0].boxplot([df.loc[df.OverallQual == q, "SalePrice"].values for q in qualites],
                        positions=qualites, widths=0.62, **STYLE_BOITE)
colorer(bp, len(qualites))
axes[0, 0].set_xlabel("OverallQual (note de 1 a 10)")
axes[0, 0].set_ylabel("Prix de vente")
viz.format_dollars(axes[0, 0])
viz.titrer(axes[0, 0], "(a) Qualite generale",
           "Croissance acceleree : +1 point vaut de plus en plus cher")

axes[0, 1].scatter(df.GrLivArea, y, s=11, color=viz.BLEU, alpha=0.45, edgecolors="none")
pente, origine = np.polyfit(df.GrLivArea, y, 1)
xs = np.linspace(df.GrLivArea.min(), df.GrLivArea.max(), 50)
axes[0, 1].plot(xs, pente * xs + origine, color=viz.ORANGE, lw=2,
                label="ajustement lineaire")
axes[0, 1].set_xlabel("Surface habitable (pi2)")
axes[0, 1].set_ylabel("Prix de vente")
axes[0, 1].legend(loc="upper left")
viz.format_dollars(axes[0, 1])
viz.titrer(axes[0, 1], "(b) Surface habitable",
           "Lineaire, mais la dispersion croit avec la surface")

axes[1, 0].scatter(df.YearBuilt, y, s=11, color=viz.BLEU, alpha=0.4, edgecolors="none",
                   label="maisons")
decennies = pd.cut(df.YearBuilt, bins=range(1870, 2021, 10))
med = df.groupby(decennies, observed=False)["SalePrice"].median().dropna()
axes[1, 0].plot([iv.mid for iv in med.index], med.values, color=viz.ORANGE, lw=2.2,
                marker="o", ms=4, label="mediane par decennie")
axes[1, 0].set_xlabel("Annee de construction")
axes[1, 0].set_ylabel("Prix de vente")
axes[1, 0].legend(loc="upper left")
viz.format_dollars(axes[1, 0])
viz.titrer(axes[1, 0], "(c) Annee de construction",
           "Non lineaire : plateau 1900-1950 puis forte hausse")

axes[1, 1].scatter(df.TotalBsmtSF, y, s=11, color=viz.BLEU, alpha=0.4, edgecolors="none")
axes[1, 1].axvline(0, color=viz.ROUGE, lw=2, ls="--",
                   label=f"{(df.TotalBsmtSF == 0).sum()} maisons sans sous-sol")
axes[1, 1].set_xlabel("Surface du sous-sol (pi2)")
axes[1, 1].set_ylabel("Prix de vente")
axes[1, 1].legend(loc="upper left")
viz.format_dollars(axes[1, 1])
viz.titrer(axes[1, 1], "(d) Surface du sous-sol",
           "Masse de points en zero : absence, pas valeur nulle")

fig.tight_layout()
viz.sauver(fig, "fig_04_relations_cle")
plt.show()
"""),

md("""
**(a) `OverallQual` est convexe.** Passer de 5 à 6 ajoute environ 27 000 \\$, de 9 à 10
environ 87 000. Une régression linéaire, qui impose un effet constant, ne peut pas le
représenter.

**(b) `GrLivArea` est linéaire mais hétéroscédastique** : le nuage s'évase, l'erreur est
plus grande sur les grandes maisons. C'est ce que le log corrige.

**(c) `YearBuilt` est franchement non linéaire** : plateau 1900-1950 puis forte hausse. Le
$r$ de 0,587 sous-estime son pouvoir prédictif.

**(d) `TotalBsmtSF` est à inflation de zéros** : les maisons sans sous-sol forment une
colonne qui ne suit pas la tendance. D'où le besoin d'une binaire.

> **Trois relations sur quatre sont non linéaires.** C'est l'argument central en faveur des
> forêts de décision : elles capturent ces courbures sans qu'on ait à les spécifier. Nous
> gardons la régression linéaire comme référence, pour quantifier ce que les modèles non
> linéaires apportent.
"""),

md("""
## 2.5 Variables catégorielles
"""),

code("""
# Effet du quartier (25 modalites)
ordre = df.groupby("Neighborhood")["SalePrice"].median().sort_values().index
effectifs = df.Neighborhood.value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot([df.loc[df.Neighborhood == q, "SalePrice"].values for q in ordre],
                positions=range(len(ordre)), widths=0.66, **STYLE_BOITE)
colorer(bp, len(ordre))
ax.set_xticks(range(len(ordre)), [f"{q}\\n(n={effectifs[q]})" for q in ordre],
              rotation=90, fontsize=7.5)
ax.set_ylabel("Prix de vente")
viz.format_dollars(ax)
ax.axhline(y.median(), color=viz.ORANGE, lw=1.8, ls="--",
           label=f"mediane globale {y.median():,.0f} $")
ax.legend(loc="upper left")
viz.titrer(ax, "Prix par quartier, tries par prix median croissant",
           "Du simple au triple entre les quartiers extremes")
fig.tight_layout()
viz.sauver(fig, "fig_05_quartiers")
plt.show()

recap_q = df.groupby("Neighborhood")["SalePrice"].agg(["count", "median"]).sort_values("median")
display(pd.concat([recap_q.head(3), recap_q.tail(3)]))
print(f"Rapport entre extremes : x{recap_q['median'].iloc[-1] / recap_q['median'].iloc[0]:.2f}")
"""),

code("""
# Trois variables qualitatives
ECHELLES = {
    "ExterQual": (["Fa", "TA", "Gd", "Ex"], "Qualite du revetement exterieur"),
    "KitchenQual": (["Fa", "TA", "Gd", "Ex"], "Qualite de la cuisine"),
    "CentralAir": (["N", "Y"], "Climatisation centrale"),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (col, (ordre_mod, libelle)) in zip(axes, ECHELLES.items()):
    mods = [m for m in ordre_mod if m in df[col].unique()]
    bp = ax.boxplot([df.loc[df[col] == m, "SalePrice"].values for m in mods],
                    positions=range(len(mods)), widths=0.6, **STYLE_BOITE)
    colorer(bp, len(mods))
    ax.set_xticks(range(len(mods)), [f"{m}\\n(n={(df[col] == m).sum()})" for m in mods],
                  fontsize=9)
    ax.set_ylabel("Prix de vente")
    viz.format_dollars(ax)
    med = df.groupby(col)["SalePrice"].median()
    viz.titrer(ax, libelle, f"rapport extreme : x{med[mods[-1]] / med[mods[0]]:.1f}")

fig.tight_layout()
viz.sauver(fig, "fig_06_variables_qualitatives")
plt.show()
"""),

md("""
**Le quartier est un prédicteur de premier ordre** : de 88 000 \\$ (`MeadowV`) à
315 000 \\$ (`NridgHt`), soit un rapport de **3,6**. Aucune variable numérique ne produit
un tel écart. Deux réserves : effectifs très inégaux (17 maisons contre 225), et 25
modalités en one-hot pour 1 460 lignes. TFDF évite ce problème.

**Les variables de qualité sont très discriminantes** : `ExterQual` x4,4, `KitchenQual`
x2,8, `CentralAir` x1,7 entre modalités extrêmes.

> **Décision** : ces variables sont **ordinales** (`Fa < TA < Gd < Ex`). Un one-hot
> détruirait cette information. Encodage ordinal à l'étape 4, après vérification de
> monotonie.
"""),

md("""
## 2.6 Valeurs manquantes et distributions

La question n'est pas "combien de trous" mais **"pourquoi ce trou existe"**.
"""),

code("""
# Classification selon la NATURE du trou
ABSENCE_EQUIPEMENT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
]
nb_nan = df.isna().sum()
nb_nan = nb_nan[nb_nan > 0].sort_values()
est_absence = [c in ABSENCE_EQUIPEMENT for c in nb_nan.index]

fig, ax = plt.subplots(figsize=(9.5, 6))
barres = ax.barh(nb_nan.index, 100 * nb_nan.values / len(df), height=0.7,
                 color=[viz.BLEU if a else viz.ORANGE for a in est_absence])
ax.bar_label(barres, fmt="%.1f%%", padding=4, fontsize=8.5, color=viz.ENCRE_2)
ax.set_xlabel("Part de valeurs manquantes (%)")
ax.set_xlim(0, 112)
ax.grid(True, axis="x"); ax.grid(False, axis="y")
ax.legend(handles=[Patch(color=viz.BLEU, label="absence d'equipement (remplir par 'Absent')"),
                   Patch(color=viz.ORANGE, label="donnee reellement manquante (imputer)")],
          loc="lower right")
viz.titrer(ax, "19 colonnes concernees, deux natures tres differentes",
           "Seules les 4 colonnes orange demandent une imputation")
fig.tight_layout()
viz.sauver(fig, "fig_07_valeurs_manquantes")
plt.show()
"""),

code("""
# Preuve : "NA" signifie-t-il bien "pas d'equipement" ?
# On croise la colonne qualitative avec le compteur quantitatif correspondant.
print("Concordance entre le 'NA' qualitatif et le compteur quantitatif :\\n")
for col, condition, libelle in [
        ("FireplaceQu", df.Fireplaces == 0, "sans cheminee"),
        ("GarageType", df.GarageArea == 0, "sans garage"),
        ("BsmtQual", df.TotalBsmtSF == 0, "sans sous-sol"),
        ("PoolQC", df.PoolArea == 0, "sans piscine")]:
    nan_col = df[col].isna()
    print(f"  {col:12s} {nan_col.sum():4d} NaN | {condition.sum():4d} maisons {libelle:14s}"
          f" | concordance {(nan_col == condition).mean() * 100:5.1f} %")

par_q = df.groupby("Neighborhood")["LotFrontage"].median()
print(f"\\nLotFrontage : mediane globale {df.LotFrontage.median():.0f} pieds, "
      f"medianes par quartier de {par_q.min():.0f} a {par_q.max():.0f} "
      f"(ecart-type {par_q.std():.1f})")
"""),

code("""
# Distributions des variables numeriques (analyse du notebook original)
asym = num_feats.skew().sort_values(ascending=False)
display(asym.head(8).to_frame("skewness").round(2))
print(f"Variables avec |skew| > 1 : {(asym.abs() > 1).sum()} / {len(asym)}")

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.ravel(), asym.head(8).index):
    ax.hist(df[col].dropna(), bins=40, color=viz.BLEU)
    ax.set_xlabel(col)
    ax.set_ylabel("Effectif")
    viz.titrer(ax, col, f"skew {asym[col]:.1f}, {100 * (df[col] == 0).mean():.0f} % de zeros")
fig.tight_layout()
viz.sauver(fig, "fig_08_asymetrie")
plt.show()
"""),

md("""
**Concordance de 100 % sur les quatre contrôles** : chaque `NaN` de `FireplaceQu`
correspond exactement à une maison sans cheminée, chaque `NaN` de `PoolQC` à une maison
sans piscine. C'est une preuve, pas une supposition. Ces colonnes seront remplies par
`"Absent"`, surtout pas par le mode, ce qui reviendrait à inventer des piscines.

**`LotFrontage` est le seul cas délicat** : médianes par quartier de 21 à 91 pieds,
écart-type 20,6. Les quartiers sont très hétérogènes, l'urbanisme étant homogène à
l'intérieur d'un quartier.

**Les variables les plus asymétriques** ont le même profil : `MiscVal` (skew 24,5) est à
96,4 % nulle, `PoolArea` (14,8) à 99,5 %. Ce sont deux informations mélangées, existence et
taille.

> **Décisions.** 15 colonnes "absence" vers `"Absent"` ou `0`. `LotFrontage` vers la médiane
> du quartier **apprise sur le train seul**. `MasVnrType`/`Area` et `Electrical` par le
> mode. Variables à inflation de zéros : créer une binaire de présence. `LotArea` : log,
> **pour les modèles linéaires seulement**, car une transformation monotone ne change rien
> pour un arbre.
"""),

md("""
## 2.7 Détection des anomalies

Supprimer des points est lourd : mal fait, cela revient à truquer les données. Nous
n'appliquons pas de règle automatique mais exigeons **deux conditions cumulatives** : le
point est statistiquement extrême, **et** une explication métier le rend non représentatif.
Sans la seconde, on garde : une maison chère n'est pas une erreur.
"""),

code("""
# Les maisons de tres grande surface
grandes = df[df.GrLivArea > 4000][
    ["Id", "GrLivArea", "SalePrice", "OverallQual", "SaleCondition", "Neighborhood"]]
display(grandes.sort_values("GrLivArea"))

print(f"Prix median au pi2 sur tout le jeu : "
      f"{(df.SalePrice / df.GrLivArea).median():.0f} $/pi2")
for _, r in grandes.iterrows():
    print(f"  Id {int(r.Id):4d} : {r.SalePrice / r.GrLivArea:5.0f} $/pi2  ({r.SaleCondition})")
"""),

code("""
# Ou se situent ces points, et pourquoi deux seulement sont ecartes
A_SUPPRIMER = [524, 1299]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ordinaire = ~df.Id.isin(A_SUPPRIMER)
axes[0].scatter(df.loc[ordinaire, "GrLivArea"], df.loc[ordinaire, "SalePrice"],
                s=12, color=viz.BLEU, alpha=0.4, edgecolors="none",
                label="observations conservees")
axes[0].scatter(df.loc[~ordinaire, "GrLivArea"], df.loc[~ordinaire, "SalePrice"],
                s=90, color=viz.ROUGE, edgecolors=viz.SURFACE, linewidths=2,
                zorder=5, label="observations ecartees")
for _, r in df[df.Id.isin(A_SUPPRIMER)].iterrows():
    axes[0].annotate(f"Id {int(r.Id)}", (r.GrLivArea, r.SalePrice),
                     textcoords="offset points", xytext=(-12, 14),
                     fontsize=9, color=viz.ROUGE, fontweight="bold")
axes[0].set_xlabel("Surface habitable (pi2)")
axes[0].set_ylabel("Prix de vente")
axes[0].legend(loc="upper left")
viz.format_dollars(axes[0])
viz.titrer(axes[0], "Surface habitable et prix",
           "Deux maisons tres grandes vendues a prix casse")

ordre_sc = df.groupby("SaleCondition")["SalePrice"].median().sort_values().index
bp = axes[1].boxplot([df.loc[df.SaleCondition == s, "SalePrice"].values for s in ordre_sc],
                     positions=range(len(ordre_sc)), widths=0.62, **STYLE_BOITE)
colorer(bp, len(ordre_sc))
axes[1].set_xticks(range(len(ordre_sc)),
                   [f"{s}\\n(n={(df.SaleCondition == s).sum()})" for s in ordre_sc],
                   fontsize=8.5)
axes[1].set_ylabel("Prix de vente")
viz.format_dollars(axes[1])
viz.titrer(axes[1], "Prix selon la condition de vente",
           "'Partial' = maison neuve vendue avant achevement")

fig.tight_layout()
viz.sauver(fig, "fig_09_anomalies")
plt.show()
"""),

md("""
| Id | Surface | Prix | \\$/pi² | Condition | Verdict |
|---|---|---|---|---|---|
| 692 | 4 316 | 755 000 \\$ | 175 | `Normal` | **Conservée** |
| 1183 | 4 476 | 745 000 \\$ | 166 | `Abnorml` | **Conservée** |
| **524** | 4 676 | 184 750 \\$ | **40** | `Partial` | **Écartée** |
| **1299** | 5 642 | 160 000 \\$ | **28** | `Partial` | **Écartée** |

Le prix médian du jeu est de 120 \\$/pi². Les maisons 524 et 1299 se sont vendues trois à
quatre fois moins cher que le marché, alors qu'elles sont les plus grandes du jeu et notées
`OverallQual = 10`. **L'explication est dans `SaleCondition`** : condition `Partial`, maison
neuve vendue avant achèvement. Le prix ne correspond pas à un bien fini. Cette exclusion est
recommandée par Dean De Cock, auteur du dataset (*Journal of Statistics Education*, 2011).

> **Décision** : écarter 2 observations sur 1 460 (0,14 %), sur critère double. Les maisons
> 692 et 1183 sont conservées, leur prix étant cohérent avec leur surface. La suppression
> est appliquée **au jeu d'entraînement seulement** : nettoyer la validation améliorerait
> artificiellement les métriques en retirant les cas difficiles de l'examen.
"""),

md("""
## 2.8 Les sept décisions issues de l'EDA

| # | Constat | Décision | Étape |
|---|---|---|---|
| 1 | Cible asymétrique (1,88 vers 0,12) | Entraîner sur `log(SalePrice)` | 5 |
| 2 | `OverallQual` (0,82) et `GrLivArea` (0,70) dominent | Variables pivots | 4 |
| 3 | 4 paires redondantes (jusqu'à 0,88) | Agrégats, et Ridge pour le linéaire | 4 et 5 |
| 4 | 3 relations sur 4 non linéaires | Arbres en première intention | 5 |
| 5 | Quartier : rapport x3,6 | TFDF la gère sans encodage | 4 et 5 |
| 6 | 15 colonnes `NA` = absence (concordance 100 %) | Remplir par `"Absent"` | 3 |
| 7 | 2 maisons à prix/pi² aberrant | Écarter, de l'entraînement seulement | 3 |

**Ce que cette EDA ajoute au notebook original** : validation empirique de la transformation
log, analyse de multicolinéarité, examen de la forme des relations (qui *justifie* le choix
des arbres au lieu de le postuler), preuve par les données de la sémantique des manquants,
et détection d'anomalies à critère métier explicite.

**Limites** : analyse univariée et bivariée, les interactions sont laissées aux arbres.
Pearson sous-estime les relations courbes. Aucune analyse spatiale possible, le dataset ne
fournit pas de coordonnées GPS.
"""),
    ]

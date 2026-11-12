"""Section 5 du pipeline : MACHINE LEARNING."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-5"></a>
# Étape 5 : MACHINE LEARNING

**Stratégie en quatre temps**, chaque modèle répondant à une question précise.

| # | Modèle | Question |
|---|---|---|
| 1 | TFDF Random Forest, données brutes | Reproduction du notebook original, point de référence |
| 2 | TFDF Random Forest, notre pipeline | Que valent les étapes 3 et 4 ? |
| 3 | TFDF Gradient Boosted Trees | Un autre algorithme de la même bibliothèque fait-il mieux ? |
| 4 à 6 | Ridge, Random Forest, Gradient Boosting (scikit-learn) | Comparaison inter-bibliothèques |

Nous nous arrêtons à six. Multiplier les algorithmes sans raison dilue le travail au lieu
de l'enrichir.

**Comparaison équitable** : même découpage (1 167 / 292), même graine (42), même jeu de
validation. Seules diffèrent les transformations imposées par chaque famille.
"""),

code("""
# Imports de l'etape 5
import time

import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

print(f"TensorFlow                 : {tf.__version__}")
print(f"TensorFlow Decision Forests: {tfdf.__version__}")

# Registre commun : chaque modele y depose ses predictions, l'etape 6 les evalue
RESULTATS = {}


def enregistrer(nom, famille, modele, pred_train, pred_valid, duree, commentaire=""):
    \"\"\"Stocke les predictions EN DOLLARS, quelle que soit l'echelle d'entrainement.\"\"\"
    RESULTATS[nom] = {
        "famille": famille, "modele": modele,
        "pred_train": np.asarray(pred_train, dtype=float).ravel(),
        "pred_valid": np.asarray(pred_valid, dtype=float).ravel(),
        "duree_s": round(duree, 2), "commentaire": commentaire,
    }
    rmse = np.sqrt(((RESULTATS[nom]["pred_valid"] - y_valid.values) ** 2).mean())
    print(f"  {nom:38s} RMSE validation = {rmse:>10,.0f} $   ({duree:.1f} s)")
"""),

md("""
## 5.1 Reproduction du notebook original

Le notebook de référence tient en quatre étapes : charger `train.csv`, supprimer `Id`,
convertir en `tf.data.Dataset`, entraîner un `RandomForestModel` sur le prix brut.
**Aucun prétraitement.** C'est volontaire : TFDF gère nativement les valeurs manquantes et
les catégorielles.

**Comment fonctionne une forêt aléatoire.** Un arbre découpe récursivement les données par
des tests du type `GrLivArea < 1500`, en choisissant celui qui réduit le plus la variance.
Un arbre seul surapprend. Une forêt entraîne N arbres sur des échantillons bootstrap
différents, chacun ne voyant qu'un sous-ensemble aléatoire de variables à chaque nœud, puis
moyenne. Ce double hasard décorrèle les arbres : leurs erreurs se compensent.

| Avantages | Limites |
|---|---|
| Aucun prétraitement requis | Peu interprétable |
| Capture non-linéarités et interactions | N'extrapole pas hors de la plage vue |
| Insensible aux échelles et à la multicolinéarité | Moins précis qu'un boosting bien réglé |

**Évaluation OOB** : chaque arbre n'ayant vu qu'environ 63 % des données, les 37 % restants
servent à l'évaluer. Cela donne une estimation de l'erreur de généralisation sans jeu
séparé, mais qui ne voit pas le prétraitement.
"""),

code("""
# MODELE 1 : reproduction fidele du notebook original
# Donnees brutes, aucune imputation, cible brute. Seul 'Id' est retire.
orig_train = train_brut.drop(columns=["Id"])
orig_valid = valid_brut.drop(columns=["Id"])
print(f"Valeurs manquantes laissees telles quelles : {orig_train.isna().sum().sum()}")
print("TFDF les traite nativement, sans imputation.\\n")

ds_train = tfdf.keras.pd_dataframe_to_tf_dataset(
    orig_train, label="SalePrice", task=tfdf.keras.Task.REGRESSION)
ds_valid = tfdf.keras.pd_dataframe_to_tf_dataset(
    orig_valid, label="SalePrice", task=tfdf.keras.Task.REGRESSION)

debut = time.time()
tfdf_original = tfdf.keras.RandomForestModel(
    task=tfdf.keras.Task.REGRESSION, random_seed=GRAINE, verbose=0)
tfdf_original.compile(metrics=["mse"])
tfdf_original.fit(ds_train)
duree = time.time() - debut

inspecteur = tfdf_original.make_inspector()
print(f"Arbres entraines : {inspecteur.num_trees()}")
print(f"RMSE OOB (estimee par TFDF sans jeu de validation) : "
      f"{inspecteur.evaluation().rmse:,.0f} $\\n")

enregistrer("1. TFDF RF (notebook original)", "TFDF", tfdf_original,
            tfdf_original.predict(ds_train, verbose=0),
            tfdf_original.predict(ds_valid, verbose=0),
            duree, "donnees brutes, cible brute")
"""),

md("""
## 5.2 Version améliorée

Mêmes maisons, avec le travail des étapes 3 et 4 : manquants traités sémantiquement,
2 anomalies écartées, 13 variables créées, 20 ordinales encodées, cible en log.

Comparer les modèles 1 et 2 isole exactement **l'apport de notre préparation**, puisque
l'algorithme et ses réglages sont identiques.

**Note sur la reconversion** : `expm1` donne la médiane conditionnelle, pas la moyenne, ce
qui introduit un biais à la baisse d'environ 1 %. La médiane étant plus robuste pour une
estimation de prix, nous n'appliquons pas de correction, mais nous le signalons.
"""),

code("""
# MODELE 2 : TFDF Random Forest sur notre pipeline, cible en log
# TFDF attend la cible dans le DataFrame : on l'ajoute sous un nom dedie
tr_log = X_train_arbres.copy(); tr_log["cible_log"] = y_train_log.values
va_log = X_valid_arbres.copy(); va_log["cible_log"] = y_valid_log.values

ds_train_log = tfdf.keras.pd_dataframe_to_tf_dataset(
    tr_log, label="cible_log", task=tfdf.keras.Task.REGRESSION)
ds_valid_log = tfdf.keras.pd_dataframe_to_tf_dataset(
    va_log, label="cible_log", task=tfdf.keras.Task.REGRESSION)

debut = time.time()
tfdf_rf = tfdf.keras.RandomForestModel(
    task=tfdf.keras.Task.REGRESSION,
    num_trees=300,          # au-dela, le gain devient negligeable
    max_depth=16,           # profondeur suffisante pour 91 variables
    random_seed=GRAINE, verbose=0)
tfdf_rf.compile(metrics=["mse"])
tfdf_rf.fit(ds_train_log)
duree = time.time() - debut

rmse_oob_log = tfdf_rf.make_inspector().evaluation().rmse
print(f"RMSE OOB en espace log : {rmse_oob_log:.4f} "
      f"(soit environ {100 * rmse_oob_log:.1f} % d'erreur relative)\\n")

# Reconversion en dollars : expm1 est l'inverse de log1p
enregistrer("2. TFDF RF (notre pipeline)", "TFDF", tfdf_rf,
            np.expm1(tfdf_rf.predict(ds_train_log, verbose=0)),
            np.expm1(tfdf_rf.predict(ds_valid_log, verbose=0)),
            duree, "pipeline A, cible log")
"""),

md("""
## 5.3 Gradient Boosted Trees

La forêt construit ses arbres **en parallèle** puis moyenne. Le boosting les construit **en
séquence**, chaque arbre corrigeant l'erreur résiduelle des précédents :

$$F_m(x) = F_{m-1}(x) + \\nu \\cdot h_m(x)$$

| | Forêt aléatoire | Gradient boosting |
|---|---|---|
| Construction | Parallèle | Séquentielle, corrective |
| Réduit surtout | La variance | Le biais |
| Arbres | Profonds | Peu profonds (3 à 6 niveaux) |
| Surapprentissage | Difficile | **Facile si trop d'arbres** |

TFDF contrôle ce risque par **arrêt précoce** : il réserve 10 % du train en validation
interne et stoppe quand l'erreur cesse de baisser. Le nombre d'arbres est donc choisi par
l'algorithme, pas par nous.
"""),

code("""
# MODELE 3 : TFDF Gradient Boosted Trees
debut = time.time()
tfdf_gbt = tfdf.keras.GradientBoostedTreesModel(
    task=tfdf.keras.Task.REGRESSION,
    num_trees=500,              # borne haute, l'arret precoce tranchera
    shrinkage=0.05,             # taux d'apprentissage, petit = plus prudent
    max_depth=6,
    early_stopping="LOSS_INCREASE",
    validation_ratio=0.10,      # part du train reservee au controle interne
    random_seed=GRAINE, verbose=0)
tfdf_gbt.compile(metrics=["mse"])
tfdf_gbt.fit(ds_train_log)
duree = time.time() - debut

insp_gbt = tfdf_gbt.make_inspector()
print(f"Arbres conserves par l'arret precoce : {insp_gbt.num_trees()} sur 500 autorises\\n")

enregistrer("3. TFDF Gradient Boosted Trees", "TFDF", tfdf_gbt,
            np.expm1(tfdf_gbt.predict(ds_train_log, verbose=0)),
            np.expm1(tfdf_gbt.predict(ds_valid_log, verbose=0)),
            duree, "arret precoce automatique")
"""),

md("""
## 5.4 Modèles scikit-learn

**Ridge** : régression linéaire avec pénalité $L_2$,
$\\min_\\beta \\|y - X\\beta\\|^2 + \\alpha\\|\\beta\\|^2$. Le terme de pénalité empêche
les coefficients de diverger. C'est la réponse directe aux deux problèmes identifiés : la
multicolinéarité du § 2.3 et le ratio de 4,7 observations par variable du § 4.3.

$\\alpha$ est choisi par validation croisée **à l'intérieur du jeu d'entraînement**. Le
choisir en regardant le score de validation serait une fuite, et une forme classique de
triche : on rapporterait la performance du meilleur essai au lieu de celle du modèle.

**Random Forest et Gradient Boosting sklearn** : mêmes principes qu'en § 5.1 et 5.3, mais
sur données encodées. La comparaison avec TFDF mesure l'effet du traitement natif du
catégoriel contre le one-hot.
"""),

code("""
# MODELE 4 : Ridge (alpha choisi par validation croisee INTERNE au train)
ALPHAS = np.logspace(-2, 3, 40)

debut = time.time()
ridge = RidgeCV(alphas=ALPHAS, cv=KFold(5, shuffle=True, random_state=GRAINE))
ridge.fit(X_train_lin, y_train_log)
duree = time.time() - debut

print(f"alpha retenu : {ridge.alpha_:.3f} "
      f"(explore entre {ALPHAS.min():.2f} et {ALPHAS.max():.0f})")
print(f"Coefficients non negligeables : "
      f"{(np.abs(ridge.coef_) > 1e-3).sum()} sur {len(ridge.coef_)}\\n")

enregistrer("4. Ridge (scikit-learn)", "lineaire", ridge,
            np.expm1(ridge.predict(X_train_lin)),
            np.expm1(ridge.predict(X_valid_lin)),
            duree, f"alpha = {ridge.alpha_:.2f}")
"""),

code("""
# MODELES 5 et 6 : Random Forest et Gradient Boosting scikit-learn
debut = time.time()
rf_sk = RandomForestRegressor(
    n_estimators=300,
    max_features="sqrt",      # sous-echantillonnage des variables a chaque noeud
    min_samples_leaf=1,
    n_jobs=-1, random_state=GRAINE)
rf_sk.fit(X_train_lin, y_train_log)
duree = time.time() - debut
enregistrer("5. Random Forest (scikit-learn)", "arbres sklearn", rf_sk,
            np.expm1(rf_sk.predict(X_train_lin)),
            np.expm1(rf_sk.predict(X_valid_lin)),
            duree, "300 arbres, max_features=sqrt")

debut = time.time()
gb_sk = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,            # stochastic gradient boosting, reduit la variance
    validation_fraction=0.10,
    n_iter_no_change=30,      # arret precoce, comme TFDF
    random_state=GRAINE)
gb_sk.fit(X_train_lin, y_train_log)
duree = time.time() - debut
print(f"\\n  Arbres conserves par l'arret precoce : {gb_sk.n_estimators_} sur 500")
enregistrer("6. Gradient Boosting (scikit-learn)", "arbres sklearn", gb_sk,
            np.expm1(gb_sk.predict(X_train_lin)),
            np.expm1(gb_sk.predict(X_valid_lin)),
            duree, "arret precoce")
"""),

md("""
## 5.5 Validation croisée

292 maisons, c'est peu : un écart de 2 000 \\$ entre deux modèles peut venir du hasard du
découpage. La validation croisée à 5 plis donne 5 mesures, donc une moyenne **et un
écart-type**. C'est cet écart-type qui permet de dire si une différence est significative.

Elle est faite **uniquement sur le jeu d'entraînement**.
"""),

code("""
# Validation croisee 5 plis, sur le jeu d'entrainement uniquement
decoupage = KFold(n_splits=5, shuffle=True, random_state=GRAINE)

A_VALIDER = {
    "Ridge": RidgeCV(alphas=ALPHAS),
    "Random Forest (sklearn)": RandomForestRegressor(
        n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=GRAINE),
    "Gradient Boosting (sklearn)": GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8,
        validation_fraction=0.10, n_iter_no_change=30, random_state=GRAINE),
}

lignes = []
for nom, modele in A_VALIDER.items():
    debut = time.time()
    # scoring negatif : scikit-learn maximise toujours, on prend l'oppose
    scores = -cross_val_score(modele, X_train_lin, y_train_log, cv=decoupage,
                              scoring="neg_root_mean_squared_error", n_jobs=1)
    lignes.append({
        "modele": nom,
        "RMSE_log_moyenne": round(scores.mean(), 4),
        "ecart_type": round(scores.std(), 4),
        "min": round(scores.min(), 4),
        "max": round(scores.max(), 4),
        "duree_s": round(time.time() - debut, 1),
    })
    print(f"  {nom:30s} {scores.mean():.4f} +/- {scores.std():.4f}")

cv_resultats = pd.DataFrame(lignes).sort_values("RMSE_log_moyenne")
display(cv_resultats)
"""),

code("""
# Validation croisee pour TFDF : l'API Keras n'est pas compatible avec
# cross_val_score, on ecrit donc la boucle a la main.
def cv_tfdf(constructeur_modele, X, y_log, n_plis=5):
    \"\"\"RMSE en espace log sur chaque pli, pour un modele TFDF.\"\"\"
    scores = []
    for i, (idx_tr, idx_va) in enumerate(KFold(n_plis, shuffle=True,
                                               random_state=GRAINE).split(X)):
        d_tr = X.iloc[idx_tr].copy(); d_tr["cible_log"] = y_log.iloc[idx_tr].values
        d_va = X.iloc[idx_va].copy(); d_va["cible_log"] = y_log.iloc[idx_va].values
        m = constructeur_modele()
        m.fit(tfdf.keras.pd_dataframe_to_tf_dataset(
            d_tr, label="cible_log", task=tfdf.keras.Task.REGRESSION))
        pred = m.predict(tfdf.keras.pd_dataframe_to_tf_dataset(
            d_va, label="cible_log", task=tfdf.keras.Task.REGRESSION),
            verbose=0).ravel()
        scores.append(float(np.sqrt(((pred - y_log.iloc[idx_va].values) ** 2).mean())))
    return np.array(scores)


for nom, fabrique in [
    ("TFDF Random Forest", lambda: tfdf.keras.RandomForestModel(
        task=tfdf.keras.Task.REGRESSION, num_trees=300, max_depth=16,
        random_seed=GRAINE, verbose=0)),
    ("TFDF Gradient Boosted", lambda: tfdf.keras.GradientBoostedTreesModel(
        task=tfdf.keras.Task.REGRESSION, num_trees=500, shrinkage=0.05, max_depth=6,
        early_stopping="LOSS_INCREASE", validation_ratio=0.10,
        random_seed=GRAINE, verbose=0)),
]:
    debut = time.time()
    scores = cv_tfdf(fabrique, X_train_arbres, y_train_log)
    cv_resultats = pd.concat([cv_resultats, pd.DataFrame([{
        "modele": nom,
        "RMSE_log_moyenne": round(scores.mean(), 4),
        "ecart_type": round(scores.std(), 4),
        "min": round(scores.min(), 4), "max": round(scores.max(), 4),
        "duree_s": round(time.time() - debut, 1),
    }])], ignore_index=True)
    print(f"  {nom:30s} {scores.mean():.4f} +/- {scores.std():.4f}")

cv_resultats = cv_resultats.sort_values("RMSE_log_moyenne").reset_index(drop=True)
display(cv_resultats)
cv_resultats.to_csv(REPORTS / "validation_croisee.csv", index=False)
"""),

code("""
# Visualisation : une barre d'erreur qui se chevauche = difference non prouvee
ordre_cv = cv_resultats.sort_values("RMSE_log_moyenne", ascending=False)

fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.barh(ordre_cv.modele, ordre_cv.RMSE_log_moyenne,
        xerr=ordre_cv.ecart_type, height=0.6, color=viz.BLEU,
        error_kw=dict(ecolor=viz.ENCRE_2, capsize=4, lw=1.4))
for i, (_, r) in enumerate(ordre_cv.iterrows()):
    ax.text(r.RMSE_log_moyenne + r.ecart_type + 0.004, i,
            f"{r.RMSE_log_moyenne:.4f}", va="center", fontsize=9, color=viz.ENCRE_2)
ax.set_xlabel("RMSE en espace log (plus bas = meilleur)")
ax.set_xlim(0, ordre_cv.RMSE_log_moyenne.max() * 1.35)
ax.grid(True, axis="x"); ax.grid(False, axis="y")
viz.titrer(ax, "Validation croisee 5 plis sur le jeu d'entrainement",
           "Les barres montrent l'ecart-type entre plis")
fig.tight_layout()
viz.sauver(fig, "fig_12_validation_croisee")
plt.show()
"""),

md("""
## 5.6 Importance des variables

Deux mesures qui ne disent pas la même chose. **`NUM_AS_ROOT`** (TFDF) compte combien de
fois la variable sert de premier test d'un arbre, donc son pouvoir de séparation global.
**L'importance par réduction d'impureté** (scikit-learn) somme les réductions de variance
sur tous les nœuds ; elle a un biais connu, elle favorise les variables à nombreuses
modalités.
"""),

code("""
# Importances, cote TFDF et cote scikit-learn
vi = tfdf_rf.make_inspector().variable_importances()["NUM_AS_ROOT"][:15]
imp_tfdf = pd.Series({v[0].name: v[1] for v in vi}).sort_values()

imp_sk = (pd.Series(gb_sk.feature_importances_, index=noms_lin)
          .sort_values(ascending=False).head(15).sort_values())

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
axes[0].barh(imp_tfdf.index, imp_tfdf.values, color=viz.BLEU, height=0.68)
axes[0].set_xlabel("Nombre d'arbres ou la variable est a la racine")
axes[0].grid(True, axis="x"); axes[0].grid(False, axis="y")
viz.titrer(axes[0], "TFDF Random Forest", "Mesure NUM_AS_ROOT, sur 300 arbres")

axes[1].barh(imp_sk.index, 100 * imp_sk.values, color=viz.AQUA, height=0.68)
axes[1].set_xlabel("Importance (% de la reduction de variance totale)")
axes[1].grid(True, axis="x"); axes[1].grid(False, axis="y")
viz.titrer(axes[1], "Gradient Boosting scikit-learn", "Reduction d'impurete")

fig.tight_layout()
viz.sauver(fig, "fig_13_importances")
plt.show()

creees_dans_top = [v for v in imp_tfdf.index if v in ConstructeurVariables.nouvelles()]
print(f"Variables creees a l'etape 4 presentes dans le top 15 TFDF : {creees_dans_top}")
"""),

md("""
## 5.7 Sauvegarde des modèles
"""),

code("""
# Sauvegarde : format SavedModel pour TFDF, joblib pour scikit-learn
import tf_keras

tfdf_rf.save(str(MODELS / "tfdf_random_forest"))
tfdf_gbt.save(str(MODELS / "tfdf_gradient_boosted"))
joblib.dump(ridge, MODELS / "ridge.joblib")
joblib.dump(rf_sk, MODELS / "random_forest_sklearn.joblib")
joblib.dump(gb_sk, MODELS / "gradient_boosting_sklearn.joblib")

# Les predictions alimentent l'etape 6
pd.DataFrame({nom: r["pred_valid"] for nom, r in RESULTATS.items()}
             ).assign(reel=y_valid.values).to_csv(
    DATA_PROC / "predictions_validation.csv", index=False)
pd.DataFrame({nom: r["pred_train"] for nom, r in RESULTATS.items()}
             ).assign(reel=y_train.values).to_csv(
    DATA_PROC / "predictions_entrainement.csv", index=False)

print("Modeles sauvegardes :")
for p in sorted(MODELS.iterdir()):
    taille = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
              if p.is_dir() else p.stat().st_size)
    print(f"  {p.name:32s} {taille / 1024:9,.1f} Ko")

# Controle : un modele TFDF recharge doit reproduire ses predictions
recharge = tf_keras.models.load_model(str(MODELS / "tfdf_random_forest"))
pred_recharge = np.expm1(recharge.predict(ds_valid_log, verbose=0)).ravel()
assert np.allclose(pred_recharge, RESULTATS["2. TFDF RF (notre pipeline)"]["pred_valid"])
print("\\nControle : le modele TFDF recharge reproduit ses predictions a l'identique.")
"""),

md("""
## 5.8 Choix de réglage

| Réglage | Valeur | Raison |
|---|---|---|
| `num_trees` (forêts) | 300 | Au-delà, le gain devient négligeable |
| `shrinkage` (boosting) | 0,05 | Petit pas, compensé par plus d'arbres |
| `max_depth` (boosting) | 3 à 6 | Le boosting combine des apprenants faibles |
| Arrêt précoce | Activé | L'algorithme choisit le nombre d'arbres |
| `alpha` (Ridge) | Validation croisée interne | Jamais réglé sur le jeu de validation |

**Limites** : aucune recherche d'hyperparamètres exhaustive, les valeurs viennent des
recommandations usuelles. Sur 1 167 lignes, une recherche par grille risquerait de
surapprendre le protocole de validation. Pas d'empilement de modèles non plus, qui
améliorerait sans doute le score mais rendrait l'analyse comparative illisible.
"""),
    ]

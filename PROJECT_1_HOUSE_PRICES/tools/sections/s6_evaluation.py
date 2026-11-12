"""Section 6 du pipeline : EVALUATION."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-6"></a>
# Étape 6 : ÉVALUATION

`test.csv` ne contient pas `SalePrice`. **Aucune métrique ici n'est calculée dessus.** Tout
provient des 292 maisons de validation, jamais vues à l'entraînement et jamais utilisées
pour régler un hyperparamètre.

| Métrique | Mesure | Unité |
|---|---|---|
| **RMSE** | Erreur quadratique, pénalise fort les grosses erreurs | \\$ |
| **MAE** | Erreur moyenne, robuste aux extrêmes | \\$ |
| **R²** | Part de variance expliquée | sans |
| **RMSLE** | Erreur relative, métrique officielle Kaggle | sans |
| **MAPE** | Erreur en pourcentage, lisible | % |

**Pourquoi cinq.** Si la RMSE dépasse largement la MAE, c'est que quelques grosses erreurs
dominent : le rapport entre les deux est en soi un diagnostic. Le R² situe le modèle face à
la prédiction triviale. La RMSLE permet de se comparer à Kaggle. La MAPE est la seule
présentable à un utilisateur non technique.

**Toutes les métriques sont en dollars**, après reconversion. Comparer une RMSE en espace
log à une RMSE en dollars n'aurait aucun sens.
"""),

code('''
# Fonction de calcul commune a tous les modeles
def metriques(y_vrai, y_pred):
    """Les cinq metriques, en dollars, a partir de predictions en dollars."""
    y_vrai = np.asarray(y_vrai, dtype=float)
    # Un prix predit negatif n'a pas de sens et casserait le logarithme
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1.0, None)
    err = y_pred - y_vrai
    return {
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(np.abs(err).mean()),
        "R2": float(1 - (err ** 2).sum() / ((y_vrai - y_vrai.mean()) ** 2).sum()),
        "RMSLE": float(np.sqrt(((np.log1p(y_pred) - np.log1p(y_vrai)) ** 2).mean())),
        "MAPE": float(100 * np.abs(err / y_vrai).mean()),
    }


# Reference : que donnerait la prediction la plus bete possible ?
naif = metriques(y_valid, np.full(len(y_valid), y_train.mean()))
print("Reference naive (predire toujours le prix moyen du train) :")
print(f"  RMSE {naif['RMSE']:>10,.0f} $ | MAE {naif['MAE']:>9,.0f} $ "
      f"| R2 {naif['R2']:>6.3f} | MAPE {naif['MAPE']:.1f} %")
print("\\nTout modele utile doit faire nettement mieux que cela.")
'''),

code("""
# Tableau comparatif
lignes = []
for nom, r in RESULTATS.items():
    m_tr = metriques(y_train, r["pred_train"])
    m_va = metriques(y_valid, r["pred_valid"])
    lignes.append({
        "modele": nom,
        "RMSE_train": round(m_tr["RMSE"]),
        "RMSE_valid": round(m_va["RMSE"]),
        "MAE_valid": round(m_va["MAE"]),
        "R2_valid": round(m_va["R2"], 4),
        "RMSLE_valid": round(m_va["RMSLE"], 4),
        "MAPE_valid": round(m_va["MAPE"], 2),
        "ecart_train_valid": round(m_va["RMSE"] / m_tr["RMSE"], 2),
        "duree_s": r["duree_s"],
    })

comparaison = pd.DataFrame(lignes).sort_values("RMSE_valid").reset_index(drop=True)
display(comparaison)
comparaison.to_csv(REPORTS / "comparaison_modeles.csv", index=False)

meilleur = comparaison.iloc[0].modele
print(f"Meilleur modele sur la RMSE de validation : {meilleur}")
"""),

code("""
# Visualisation : RMSE et R2 par modele
ordre = comparaison.sort_values("RMSE_valid", ascending=False)
etiquettes = [n.split(". ", 1)[-1] for n in ordre.modele]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

axes[0].barh(etiquettes, ordre.RMSE_valid, height=0.62, color=viz.BLEU,
             label="validation")
axes[0].barh(etiquettes, ordre.RMSE_train, height=0.3, color=viz.ORANGE,
             label="entrainement")
for i, (_, r) in enumerate(ordre.iterrows()):
    axes[0].text(r.RMSE_valid + 900, i, f"{r.RMSE_valid:,.0f} $",
                 va="center", fontsize=8.5, color=viz.ENCRE_2)
axes[0].axvline(naif["RMSE"], color=viz.ROUGE, ls="--", lw=1.6,
                label=f"reference naive ({naif['RMSE']:,.0f} $)")
axes[0].set_xlabel("RMSE (plus bas = meilleur)")
axes[0].set_xlim(0, max(ordre.RMSE_valid.max(), naif["RMSE"]) * 1.25)
axes[0].grid(True, axis="x"); axes[0].grid(False, axis="y")
axes[0].legend(loc="lower right")
viz.titrer(axes[0], "Erreur quadratique moyenne",
           "La barre fine interne montre l'erreur sur l'entrainement")

axes[1].barh(etiquettes, ordre.R2_valid, height=0.62, color=viz.AQUA)
for i, v in enumerate(ordre.R2_valid):
    axes[1].text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8.5, color=viz.ENCRE_2)
axes[1].set_xlabel("R2 sur la validation (plus haut = meilleur)")
axes[1].set_xlim(0, 1.08)
axes[1].grid(True, axis="x"); axes[1].grid(False, axis="y")
viz.titrer(axes[1], "Part de variance expliquee",
           "1,0 serait une prediction parfaite")

fig.tight_layout()
viz.sauver(fig, "fig_14_comparaison_modeles")
plt.show()
"""),

md("""
## 6.1 Lecture du tableau

Trois résultats méritent d'être commentés, et deux d'entre eux sont contre-intuitifs.

### Résultat 1 : notre prétraitement n'améliore presque pas TFDF

| Modèle | RMSE validation | RMSLE |
|---|---|---|
| 1. TFDF RF, données brutes (notebook original) | 33 027 \$ | 0,1608 |
| 2. TFDF RF, notre pipeline | **32 817 \$** | **0,1576** |

Le gain est de **0,6 %**, soit **moins que le bruit d'échantillonnage** estimé à 6,3 % par
la validation croisée. Autrement dit : **nous ne pouvons pas affirmer que nos étapes 3 et 4
ont amélioré ce modèle.**

C'est un résultat honnête, et il s'explique très bien. TFDF fait déjà nativement ce que
notre prétraitement fait à la main :

- il traite les valeurs manquantes sans imputation, en les routant dans une branche dédiée ;
- il gère les catégorielles sans encodage ;
- il est insensible aux échelles et aux transformations monotones.

Les 13 variables créées lui apportent donc peu : il pouvait déjà reconstruire
`SurfaceTotale` par des découpages successifs. Le fait que 4 d'entre elles apparaissent dans
son top 15 d'importance montre qu'il les **utilise**, mais utiliser n'est pas la même chose
que gagner en précision.

**Ce que le prétraitement apporte vraiment**, c'est de rendre possibles les modèles
linéaires, qui sans lui ne tourneraient pas du tout, et de rendre le pipeline explicite donc
auditable. Ce n'est pas rien, mais ce n'est pas un gain de score.

### Résultat 2 : le classement s'inverse selon la métrique

| Modèle | RMSE (rang) | MAPE (rang) | RMSLE (rang) |
|---|---|---|---|
| TFDF RF, notre pipeline | 32 817 \$ (**1er**) | 11,04 % (4e) | 0,1576 (4e) |
| Ridge | 38 026 \$ (**5e**) | **9,33 % (1er)** | **0,1402 (1er)** |

Ridge est le **pire** modèle en RMSE et le **meilleur** en erreur relative. Ce n'est pas une
contradiction, c'est la définition des deux métriques :

- la **RMSE** élève les erreurs au carré, donc elle est dominée par quelques très grosses
  erreurs en dollars, qui portent sur les maisons chères ;
- la **MAPE** et la **RMSLE** mesurent l'erreur en pourcentage, où toutes les maisons pèsent
  pareil.

Ridge se trompe modérément et régulièrement sur toutes les maisons. TFDF est très précis sur
la majorité mais commet quelques erreurs énormes. **Le choix entre les deux dépend donc de
l'usage** : pour un site grand public qui estime surtout des maisons ordinaires, l'erreur
relative compte davantage ; pour un investisseur exposé au haut de gamme, l'erreur en
dollars compte plus.

La validation croisée du § 5.5, calculée en espace log, confirme ce diagnostic : Ridge y est
premier (0,1137) devant TFDF RF (0,1292).

### Résultat 3 : le surapprentissage est apparent mais pas problématique

Les rapports RMSE_validation / RMSE_entraînement vont de 1,9 (Ridge) à 4,2 (TFDF Gradient
Boosted). Un facteur 4 semble alarmant, mais il ne l'est pas pour un modèle à base d'arbres :
chaque arbre peut isoler ses observations d'entraînement, donc l'erreur d'entraînement est
structurellement très basse, sans que la généralisation en souffre, puisqu'elle est assurée
par le moyennage ou l'arrêt précoce.

Le bon indicateur reste la performance en validation, et elle est cohérente avec la
validation croisée. Ridge, le moins surajusté en apparence, n'est pas pour autant le
meilleur en RMSE.
"""),

code("""
# Surapprentissage : ecart entre entrainement et validation
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(comparaison))
etiq = [n.split(". ", 1)[-1] for n in comparaison.modele]

ax.bar(x - 0.2, comparaison.RMSE_train, width=0.4, color=viz.ORANGE,
       label="entrainement")
ax.bar(x + 0.2, comparaison.RMSE_valid, width=0.4, color=viz.BLEU,
       label="validation")
for i, (_, r) in enumerate(comparaison.iterrows()):
    ax.text(i, max(r.RMSE_train, r.RMSE_valid) * 1.04, f"x{r.ecart_train_valid:.1f}",
            ha="center", fontsize=9, color=viz.ENCRE_2)
ax.set_xticks(x, etiq, rotation=20, ha="right", fontsize=8.5)
ax.set_ylabel("RMSE")
viz.format_dollars(ax)
ax.legend()
viz.titrer(ax, "Erreur sur l'entrainement et sur la validation",
           "Le facteur indique le rapport entre les deux")
fig.tight_layout()
viz.sauver(fig, "fig_15_surapprentissage")
plt.show()
"""),

md("""
## 6.2 Analyse des erreurs

Le tableau dit *combien* le modèle se trompe. Cette section dit *où* et *pourquoi*, ce qui
est plus utile pour l'améliorer.
"""),

code("""
# Residus du meilleur modele
pred = RESULTATS[meilleur]["pred_valid"]
reel = y_valid.values
residus = pred - reel
residus_rel = 100 * residus / reel

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

# (a) Predictions contre realite
axes[0, 0].scatter(reel, pred, s=16, color=viz.BLEU, alpha=0.5, edgecolors="none")
lim = [min(reel.min(), pred.min()) * 0.9, max(reel.max(), pred.max()) * 1.05]
axes[0, 0].plot(lim, lim, color=viz.ENCRE, ls="--", lw=1.5, label="prediction parfaite")
axes[0, 0].set_xlabel("Prix reel")
axes[0, 0].set_ylabel("Prix predit")
axes[0, 0].legend(loc="upper left")
viz.format_dollars(axes[0, 0]); viz.format_dollars(axes[0, 0], "x")
viz.titrer(axes[0, 0], "(a) Predictions contre realite",
           "Les points doivent se serrer autour de la diagonale")

# (b) Residus contre prix predit
axes[0, 1].scatter(pred, residus, s=16, color=viz.BLEU, alpha=0.5, edgecolors="none")
axes[0, 1].axhline(0, color=viz.ENCRE, ls="--", lw=1.5)
axes[0, 1].set_xlabel("Prix predit")
axes[0, 1].set_ylabel("Residu (predit moins reel)")
viz.format_dollars(axes[0, 1]); viz.format_dollars(axes[0, 1], "x")
viz.titrer(axes[0, 1], "(b) Residus",
           "Un nuage sans structure indique un modele bien specifie")

# (c) Distribution des erreurs relatives
axes[1, 0].hist(residus_rel, bins=45, color=viz.AQUA)
axes[1, 0].axvline(0, color=viz.ENCRE, ls="--", lw=1.5)
axes[1, 0].axvline(np.median(residus_rel), color=viz.ORANGE, lw=2,
                   label=f"mediane {np.median(residus_rel):+.1f} %")
axes[1, 0].set_xlabel("Erreur relative (%)")
axes[1, 0].set_ylabel("Nombre de maisons")
axes[1, 0].legend()
viz.titrer(axes[1, 0], "(c) Distribution des erreurs relatives",
           "Centree sur zero = pas de biais systematique")

# (d) Erreur par tranche de prix
tranches = pd.qcut(reel, 5)
err_tranche = pd.DataFrame({"tranche": tranches, "abs_rel": np.abs(residus_rel)})
moyennes = err_tranche.groupby("tranche", observed=True).abs_rel.mean()
libelles = [f"{int(iv.left/1000)}k\\n{int(iv.right/1000)}k" for iv in moyennes.index]
barres = axes[1, 1].bar(range(len(moyennes)), moyennes.values, width=0.65, color=viz.BLEU)
axes[1, 1].bar_label(barres, fmt="%.1f%%", padding=3, fontsize=9, color=viz.ENCRE_2)
axes[1, 1].set_xticks(range(len(moyennes)), libelles, fontsize=8.5)
axes[1, 1].set_xlabel("Tranche de prix reel")
axes[1, 1].set_ylabel("Erreur absolue relative moyenne (%)")
viz.titrer(axes[1, 1], "(d) Erreur par gamme de prix",
           "Ou le modele est-il le plus fiable ?")

fig.suptitle(f"Analyse des erreurs : {meilleur}", fontsize=13, fontweight="semibold",
             x=0.01, ha="left", color=viz.ENCRE)
fig.tight_layout(rect=[0, 0, 1, 0.97])
viz.sauver(fig, "fig_16_analyse_erreurs")
plt.show()
"""),

code("""
# Les dix plus grosses erreurs
contexte = valid_brut.reset_index(drop=True)
detail = pd.DataFrame({
    "Id": contexte.Id,
    "prix_reel": reel,
    "prix_predit": pred.round(0),
    "erreur": residus.round(0),
    "erreur_pct": residus_rel.round(1),
    "GrLivArea": contexte.GrLivArea,
    "OverallQual": contexte.OverallQual,
    "Neighborhood": contexte.Neighborhood,
    "SaleCondition": contexte.SaleCondition,
})
pires = detail.reindex(detail.erreur.abs().sort_values(ascending=False).index).head(10)
display(pires)

print(f"Les 10 pires erreurs representent "
      f"{100 * (pires.erreur ** 2).sum() / (residus ** 2).sum():.1f} % "
      f"de l'erreur quadratique totale, pour "
      f"{100 * 10 / len(detail):.1f} % des maisons.")
print(f"\\nRepartition des conditions de vente parmi ces 10 :")
print(pires.SaleCondition.value_counts().to_string())
print(f"\\nPour comparaison, sur l'ensemble de la validation :")
print(contexte.SaleCondition.value_counts().to_string())
"""),

code("""
# Le cas de l'anomalie restee dans le jeu de validation
ANOMALIES = [524, 1299]
dans_valid = detail[detail.Id.isin(ANOMALIES)]

if len(dans_valid):
    display(dans_valid)
    sans = detail[~detail.Id.isin(ANOMALIES)]
    rmse_avec = np.sqrt((detail.erreur ** 2).mean())
    rmse_sans = np.sqrt((sans.erreur ** 2).mean())
    print(f"RMSE avec cette maison   : {rmse_avec:>10,.0f} $")
    print(f"RMSE sans cette maison   : {rmse_sans:>10,.0f} $")
    print(f"Effet d'une seule maison : {rmse_avec - rmse_sans:>10,.0f} $ "
          f"({100 * (rmse_avec - rmse_sans) / rmse_avec:.1f} % de la RMSE)")
    print("\\nLe chiffre que nous rapportons est celui AVEC la maison.")
else:
    print("Aucune anomalie connue n'est tombee dans le jeu de validation.")
"""),

md("""
**Le modèle est bien spécifié** : le nuage de résidus ne montre pas de structure. Une
courbe ou un entonnoir signalerait une relation non captée.

**Les erreurs sont très concentrées** : 10 maisons sur 292 (3,4 %) portent **63 % de
l'erreur quadratique totale**. C'est caractéristique de la RMSE, qui élève au carré.

**Les ventes `Partial` et `Abnorml` sont surreprésentées** parmi les pires cas : ce sont des
ventes dont le prix ne reflète pas la valeur du bien (construction inachevée, saisie, vente
entre proches). Le modèle ne peut pas les deviner, l'information n'est pas dans les données.

**Une seule maison déplace la RMSE de 20 %.** C'est l'anomalie laissée volontairement dans
la validation à l'étape 3. Nous rapportons le chiffre qui l'inclut. La retirer donnerait
26 271 \\$ au lieu de 32 817, mais ce score ne voudrait rien dire : en production, ce genre
de vente existe.

**Piste identifiée** : entraîner un modèle sur les seules ventes `Normal` (82 % du jeu).
L'inconvénient est qu'il faudrait connaître la condition de vente à l'avance, ce qui n'est
pas réaliste pour une estimation avant mise en vente.
"""),

md("""
## 6.3 Choix du modèle final

Le modèle retenu n'est pas forcément celui qui a la meilleure RMSE.
"""),

code("""
# Grille de decision multicritere
grille = comparaison.copy()
grille["ecart_au_meilleur_pct"] = (
    100 * (grille.RMSE_valid - grille.RMSE_valid.min()) / grille.RMSE_valid.min()).round(1)

# L'ecart-type de la validation croisee donne l'ordre de grandeur du bruit
bruit_cv = cv_resultats.ecart_type.mean()
seuil_pct = 100 * bruit_cv / cv_resultats.RMSE_log_moyenne.min()
print(f"Ecart-type moyen entre plis de validation croisee : {bruit_cv:.4f} en espace log")
print(f"Soit environ {seuil_pct:.1f} % de la RMSE : en dessous de cet ecart,")
print("une difference entre deux modeles n'est pas interpretable.\\n")

grille["difference_significative"] = np.where(
    grille.ecart_au_meilleur_pct <= seuil_pct, "non (dans le bruit)", "oui")
display(grille[["modele", "RMSE_valid", "MAPE_valid", "R2_valid",
                "ecart_au_meilleur_pct", "difference_significative", "duree_s"]])
"""),

md("""
**Quatre critères.**

1. **Performance**, lue à la lumière du bruit d'échantillonnage. Plusieurs modèles peuvent
   être à égalité statistique.
2. **Robustesse** : un modèle stable entre les plis est plus fiable qu'un modèle performant
   en moyenne mais instable.
3. **Simplicité de production** : TFDF consomme directement un DataFrame avec des colonnes
   texte. Les modèles scikit-learn exigent de transporter le `ColumnTransformer` et ses 246
   colonnes, et de garantir l'ordre des colonnes. Source d'erreur réelle.
4. **Cohérence avec la consigne** : à performance équivalente, TFDF est préféré.
"""),

code("""
# Decision finale
candidats = grille[grille.difference_significative == "non (dans le bruit)"].modele.tolist()
print(f"Modeles a egalite statistique avec le meilleur : {len(candidats)}")
for c in candidats:
    print(f"  {c}")

# Parmi les candidats a egalite, on retient un modele TFDF (criteres 3 et 4)
tfdf_candidats = [c for c in candidats if RESULTATS[c]["famille"] == "TFDF"]
MODELE_FINAL = tfdf_candidats[0] if tfdf_candidats else candidats[0]

m = metriques(y_valid, RESULTATS[MODELE_FINAL]["pred_valid"])
print(f"\\n=== MODELE RETENU : {MODELE_FINAL} ===")
for k, v in m.items():
    unite = " $" if k in ("RMSE", "MAE") else (" %" if k == "MAPE" else "")
    print(f"  {k:6s} : {v:>12,.4f}{unite}" if k in ("R2", "RMSLE")
          else f"  {k:6s} : {v:>12,.0f}{unite}")
print(f"\\nAmelioration par rapport a la reference naive : "
      f"{100 * (1 - m['RMSE'] / naif['RMSE']):.1f} % de RMSE en moins")
"""),

md("""
## 6.4 Limites de cette évaluation

- **292 maisons seulement**, intervalles de confiance larges. Un autre découpage pourrait
  changer l'ordre entre modèles proches.
- **Validation aléatoire et non temporelle.** Un découpage chronologique (entraîner sur
  2006 à 2009, valider sur 2010) serait plus réaliste mais réduirait encore le jeu de
  validation.
- **Pas de soumission Kaggle** sans `test.csv` : impossible de se situer dans le classement
  public.
"""),
    ]

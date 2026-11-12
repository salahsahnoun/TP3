"""Section 7 du pipeline : DEPLOIEMENT."""
from nbgen import md, code


def cells():
    return [

md("""
<a id="etape-7"></a>
# Étape 7 : DÉPLOIEMENT

**Le vrai problème du déploiement.** Une maison saisie dans une interface doit subir
**exactement** les mêmes transformations qu'une maison du jeu d'entraînement. Si l'ordre des
colonnes change, si une imputation diffère d'un dollar, le modèle reçoit une entrée qu'il
n'a jamais vue et renvoie n'importe quoi. C'est l'erreur de production la plus courante, et
elle est silencieuse : aucune exception, juste une prédiction fausse.

| Risque | Protection |
|---|---|
| Logique réécrite dans l'application | Les mêmes objets `fit`/`transform` sont rechargés |
| Code de prédiction dupliqué | Un seul module, `app/prediction.py`, utilisé par l'application **et** ce notebook |
| Dérive silencieuse | Un test compare la prédiction de l'application à celle du notebook |

**Streamlit ou FastAPI.** Streamlit livre une page web complète sans écrire d'interface,
FastAPI livre du JSON pour un autre programme. Le besoin ici est de montrer et tester le
modèle : **Streamlit**. Exposer une API demanderait une quinzaine de lignes, indiquées en
fin de section.
"""),

code("""
import json
import subprocess
import sys

sys.path.insert(0, str(APP))
print(f"Modele retenu a l'etape 6 : {MODELE_FINAL}")
"""),

md("""
## 7.1 Les artefacts à sauvegarder

Quatre choses, pas seulement le modèle : le nettoyeur (étape 3), le préprocesseur
(étape 4), le modèle (étape 5) et une configuration produite ici.

**Pourquoi des valeurs par défaut.** Demander 79 caractéristiques est irréaliste.
L'interface en demande une vingtaine et complète le reste par la médiane ou le mode,
calculés **sur le train seul** pour ne pas faire entrer d'information de la validation dans
l'application.

**Pourquoi mémoriser les types.** TFDF enregistre une signature typée : il attend `int64`
sur une quarantaine de colonnes, alors que les valeurs par défaut transitent par du JSON,
qui ne connaît que le flottant. Sans restauration explicite, le modèle refuse l'entrée. Le
même problème affecte `MSSubClass`, qui deviendrait la catégorie inédite `"50.0"` au lieu
de `"50"`. Ce genre de détail ne se découvre qu'en testant réellement la chaîne.
"""),

code("""
colonnes_brutes = [c for c in brut.columns if c not in ("Id", "SalePrice")]

# Mediane pour les numeriques, mode pour les categorielles, sur le TRAIN seul
valeurs_defaut = {}
for col in colonnes_brutes:
    serie = train_brut[col].dropna()
    if pd.api.types.is_numeric_dtype(train_brut[col]):
        valeurs_defaut[col] = float(serie.median()) if len(serie) else 0.0
    else:
        valeurs_defaut[col] = str(serie.mode()[0]) if len(serie) else "Absent"

# Modalites proposees dans les listes deroulantes de l'interface
CHAMPS_LISTE = ["Neighborhood", "MSZoning", "ExterQual", "KitchenQual", "BsmtQual",
                "HeatingQC", "CentralAir", "GarageFinish", "SaleCondition", "HouseStyle"]
modalites = {c: sorted(train_brut[c].dropna().astype(str).unique().tolist())
             for c in CHAMPS_LISTE}

print(f"{len(valeurs_defaut)} valeurs par defaut calculees sur {len(train_brut)} maisons")
print("Exemples :")
for c in ["OverallQual", "GrLivArea", "Neighborhood", "YearBuilt", "KitchenQual"]:
    print(f"   {c:16s} {valeurs_defaut[c]}")
"""),

code("""
DOSSIERS_MODELES = {
    "2. TFDF RF (notre pipeline)": "tfdf_random_forest",
    "3. TFDF Gradient Boosted Trees": "tfdf_gradient_boosted",
}
# Le modele 1 utilise les donnees brutes, il n'est pas compatible avec notre
# chaine de preparation : on retombe sur le modele 2 si besoin.
dossier_modele = DOSSIERS_MODELES.get(MODELE_FINAL, "tfdf_random_forest")
nom_deploye = next(k for k, v in DOSSIERS_MODELES.items() if v == dossier_modele)

m_final = metriques(y_valid, RESULTATS[nom_deploye]["pred_valid"])

# Les types doivent etre memorises : TFDF enregistre une signature typee, et
# une colonne passee en float64 alors que le modele attend un int64 declenche
# une erreur au moment de la prediction.
dtypes_bruts = {c: str(train_brut[c].dtype) for c in colonnes_brutes}
dtypes_prepares = {c: str(t) for c, t in X_train_arbres.dtypes.items()}

config = {
    "modele": dossier_modele,
    "nom_modele": nom_deploye.split(". ", 1)[-1],
    "colonnes_brutes": colonnes_brutes,
    "dtypes_bruts": dtypes_bruts,
    "colonnes_preparees": list(X_train_arbres.columns),
    "dtypes_prepares": dtypes_prepares,
    "valeurs_defaut": valeurs_defaut,
    "modalites": modalites,
    "n_entrainement": int(len(train_brut)),
    "n_validation": int(len(valid_brut)),
    "rmse_validation": round(m_final["RMSE"]),
    "mae_validation": round(m_final["MAE"]),
    "r2_validation": round(m_final["R2"], 4),
    "mape_validation": round(m_final["MAPE"], 2),
    "graine": GRAINE,
}
(MODELS / "deploiement.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Configuration ecrite : models/deploiement.json")
print(f"  modele deploye : {config['nom_modele']}")
print(f"  RMSE {config['rmse_validation']:,} $ | MAPE {config['mape_validation']} % "
      f"| R2 {config['r2_validation']}")
"""),

md("""
## 7.2 Test de non-régression

Le test qui compte : une maison passée par la chaîne de déploiement doit recevoir
exactement la même prédiction que dans ce notebook. Strict, à $10^{-6}$ dollar près.
"""),

code("""
from prediction import Predicteur

predicteur = Predicteur(MODELS)
print(f"Artefacts charges depuis {MODELS.relative_to(PROJET)}\\n")

# Dix maisons de validation, passees par la chaine de deploiement
# On transmet TOUTES les colonnes, valeurs manquantes comprises : c'est au
# nettoyeur de les imputer, pas aux valeurs par defaut de l'interface. Les
# remplacer ici reviendrait a tester une autre chaine que celle du notebook.
echantillon = valid_brut.head(10)
via_deploiement = []
for _, maison in echantillon.iterrows():
    saisie = {c: maison[c] for c in colonnes_brutes}
    via_deploiement.append(predicteur.predire(saisie))

via_notebook = RESULTATS[nom_deploye]["pred_valid"][:10]

verif = pd.DataFrame({
    "Id": echantillon.Id.values,
    "prix_reel": echantillon.SalePrice.values,
    "notebook": np.round(via_notebook, 2),
    "application": np.round(via_deploiement, 2),
    "ecart": np.round(np.array(via_deploiement) - via_notebook, 6),
})
display(verif)

ecart_max = float(np.abs(verif.ecart).max())
print(f"Ecart maximal : {ecart_max:.8f} $")
assert ecart_max < 1e-6, "La chaine de deploiement ne reproduit pas le notebook."
print("\\n[OK] La chaine de deploiement reproduit le notebook a l'identique.")
"""),

code("""
# Deuxieme controle : coherence des predictions sur des maisons construites
# a la main. Les prix doivent croitre avec la qualite et la surface.
scenarios = {
    "Petite maison modeste": dict(
        OverallQual=4, OverallCond=5, GrLivArea=900, TotalBsmtSF=600,
        YearBuilt=1950, GarageCars=0, GarageArea=0, FullBath=1,
        Neighborhood="MeadowV", KitchenQual="TA", ExterQual="TA"),
    "Maison moyenne": dict(
        OverallQual=6, OverallCond=5, GrLivArea=1500, TotalBsmtSF=1000,
        YearBuilt=1975, GarageCars=2, GarageArea=480, FullBath=2,
        Neighborhood="NAmes", KitchenQual="TA", ExterQual="TA"),
    "Maison recente de qualite": dict(
        OverallQual=8, OverallCond=5, GrLivArea=2200, TotalBsmtSF=1400,
        YearBuilt=2005, GarageCars=3, GarageArea=700, FullBath=2,
        Neighborhood="NridgHt", KitchenQual="Gd", ExterQual="Gd"),
    "Haut de gamme": dict(
        OverallQual=10, OverallCond=5, GrLivArea=3000, TotalBsmtSF=2000,
        YearBuilt=2008, GarageCars=3, GarageArea=850, FullBath=3,
        Neighborhood="NoRidge", KitchenQual="Ex", ExterQual="Ex"),
}

estimations = {nom: predicteur.predire(s) for nom, s in scenarios.items()}
for nom, prix in estimations.items():
    print(f"  {nom:28s} {prix:>10,.0f} $")

valeurs = list(estimations.values())
assert all(a < b for a, b in zip(valeurs, valeurs[1:])), \\
    "Les prix ne croissent pas avec la qualite : incoherence du modele."
print("\\n[OK] Les estimations croissent avec la qualite et la surface.")
print(f"Amplitude : x{valeurs[-1] / valeurs[0]:.1f} entre le scenario le plus")
print("modeste et le plus haut de gamme.")
"""),

code("""
# Troisieme controle : sensibilite a une seule variable, toutes choses egales
# par ailleurs. C'est ce que verra un utilisateur qui fait varier un curseur.
base = dict(scenarios["Maison moyenne"])
courbes = {}
for variable, valeurs_test in [
        ("GrLivArea", [800, 1200, 1600, 2000, 2400, 2800]),
        ("OverallQual", [3, 4, 5, 6, 7, 8, 9, 10])]:
    courbes[variable] = [(v, predicteur.predire({**base, variable: v}))
                         for v in valeurs_test]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (variable, points) in zip(axes, courbes.items()):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=viz.BLEU, lw=2.2, marker="o", ms=6)
    ax.set_xlabel(variable)
    ax.set_ylabel("Prix estime")
    viz.format_dollars(ax)
    viz.titrer(ax, f"Sensibilite a {variable}",
               "Toutes les autres caracteristiques sont fixees")
fig.tight_layout()
viz.sauver(fig, "fig_17_sensibilite")
plt.show()

for variable, points in courbes.items():
    croissant = all(a[1] <= b[1] for a, b in zip(points, points[1:]))
    print(f"  {variable:12s} croissance monotone : {'oui' if croissant else 'non'}")
"""),

md("""
**Le test passe à l'écart nul.** La chaîne de déploiement est identique à celle du
notebook.

**Les scénarios se comportent comme attendu** : le prix croît avec la qualité et la surface.

**Les courbes de sensibilité montrent des paliers**, et c'est normal : une forêt de décision
est une fonction **en escalier**, elle prédit par moyennes de feuilles. Conséquence pratique
à connaître : le modèle **n'extrapole pas**. Une maison de 10 000 pi², hors de la plage vue
à l'entraînement, recevra la même prédiction que la plus grande maison connue.

L'interface affiche le prix, une fourchette, et surtout **les limites du modèle**. Un outil
qui affiche un chiffre sans son incertitude donne une fausse impression de précision. La
fourchette utilise l'erreur relative moyenne, et l'interface précise que ce n'est pas un
intervalle de confiance statistique.
"""),

code("""
# Verification que les deux fichiers sont en place et syntaxiquement valides
import ast

for fichier in ["prediction.py", "app.py"]:
    chemin = APP / fichier
    source = chemin.read_text()
    ast.parse(source)
    print(f"  {fichier:16s} {len(source.splitlines()):4d} lignes   syntaxe valide")

print("\\nLancement de l'interface :")
print("    streamlit run app/app.py")
print("Puis ouvrir http://localhost:8501")
"""),

code("""
# Test de demarrage reel : on lance Streamlit en mode headless et on verifie
# que le serveur repond. C'est le seul moyen de prouver que l'application
# fonctionne vraiment, et pas seulement qu'elle compile.
import time
import urllib.request

processus = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(APP / "app.py"),
     "--server.headless=true", "--server.port=8501",
     "--browser.gatherUsageStats=false"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

code_http, erreur = None, None
try:
    for _ in range(60):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8501", timeout=2) as r:
                code_http = r.status
                break
        except Exception as e:
            erreur = e
            if processus.poll() is not None:
                break
finally:
    processus.terminate()
    try:
        processus.wait(timeout=10)
    except subprocess.TimeoutExpired:
        processus.kill()

print(f"Reponse HTTP du serveur Streamlit : {code_http}")
assert code_http == 200, f"L'application n'a pas demarre ({erreur})"
print("[OK] L'application demarre et repond. Serveur arrete apres le test.")
"""),

md("""
## 7.3 Exposer une API

La logique étant isolée dans `app/prediction.py`, il suffirait de :

```python
from fastapi import FastAPI
from pydantic import BaseModel
from app.prediction import obtenir_predicteur

api = FastAPI(title="Estimation immobiliere Ames")

class Maison(BaseModel):
    OverallQual: int
    GrLivArea: int
    Neighborhood: str
    YearBuilt: int
    # les autres champs sont facultatifs, completes par les valeurs par defaut

@api.post("/predire")
def predire(maison: Maison):
    prix = obtenir_predicteur().predire(maison.model_dump(exclude_none=True))
    return {"prix_estime": round(prix), "devise": "USD"}
```

C'est le même `Predicteur`, donc les mêmes garanties.

## 7.4 Limites du déploiement

Pas de conteneurisation, pas d'authentification, pas de surveillance de dérive, pas de
réentraînement automatique. En production réelle, il faudrait comparer la distribution des
saisies à celle du jeu d'entraînement : un modèle entraîné sur Ames 2006-2010 appliqué à
d'autres données donnerait des résultats faux en silence.
"""),
    ]

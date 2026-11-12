"""Interface d'estimation du prix d'une maison.

Lancement :
    streamlit run app/app.py

L'application ne contient aucune logique de prediction : elle appelle le module
app/prediction.py, qui est aussi celui teste dans le notebook. Une maison saisie
ici subit donc exactement le meme traitement qu'une maison du jeu d'entrainement.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction import obtenir_predicteur

st.set_page_config(page_title="Estimation de prix immobilier",
                   page_icon="maison", layout="wide")


@st.cache_resource(show_spinner="Chargement du modele...")
def charger():
    return obtenir_predicteur()


predicteur = charger()
config = predicteur.config

st.title("Estimation du prix de vente d'une maison")
st.caption(f"Modele : {config['nom_modele']}  |  "
           f"Entraine sur {config['n_entrainement']} ventes a Ames, Iowa (2006-2010)  |  "
           f"Erreur moyenne en validation : {config['mape_validation']:.1f} %")

st.info(
    "Seules les caracteristiques les plus influentes sont demandees. Les autres "
    "sont completees par la valeur mediane ou la modalite la plus frequente du "
    "jeu d'entrainement.",
    icon=None)

saisie = {}
onglet_principal, onglet_detail = st.tabs(["Caracteristiques principales",
                                           "Details complementaires"])

with onglet_principal:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Qualite et surface")
        saisie["OverallQual"] = st.slider(
            "Qualite generale des materiaux et finitions", 1, 10, 6,
            help="Note attribuee par l'evaluateur municipal. "
                 "Variable la plus predictive du modele.")
        saisie["OverallCond"] = st.slider("Etat general", 1, 10, 5)
        saisie["GrLivArea"] = st.number_input(
            "Surface habitable hors sous-sol (pi2)", 300, 6000, 1500, step=50)
        saisie["TotalBsmtSF"] = st.number_input(
            "Surface du sous-sol (pi2), 0 si absent", 0, 6000, 1000, step=50)
        saisie["1stFlrSF"] = st.number_input(
            "Surface du rez-de-chaussee (pi2)", 300, 5000, 1100, step=50)
        saisie["2ndFlrSF"] = st.number_input(
            "Surface de l'etage (pi2), 0 si absent", 0, 3000, 400, step=50)

    with c2:
        st.subheader("Localisation et terrain")
        saisie["Neighborhood"] = st.selectbox(
            "Quartier", config["modalites"]["Neighborhood"],
            index=config["modalites"]["Neighborhood"].index(
                predicteur.defauts["Neighborhood"]),
            help="Les prix medians vont du simple au triple selon le quartier.")
        saisie["MSZoning"] = st.selectbox(
            "Zonage", config["modalites"]["MSZoning"],
            index=config["modalites"]["MSZoning"].index(predicteur.defauts["MSZoning"]))
        saisie["LotArea"] = st.number_input(
            "Surface du terrain (pi2)", 1000, 250000, 9500, step=500)
        saisie["LotFrontage"] = st.number_input(
            "Facade sur rue (pieds)", 0, 350, 70, step=5)
        saisie["YearBuilt"] = st.number_input(
            "Annee de construction", 1870, 2010, 1975)
        saisie["YearRemodAdd"] = st.number_input(
            "Annee de derniere renovation", 1950, 2010, 1995)

    with c3:
        st.subheader("Equipements")
        saisie["FullBath"] = st.selectbox("Salles de bain completes", [0, 1, 2, 3, 4], 1)
        saisie["HalfBath"] = st.selectbox("Salles d'eau", [0, 1, 2], 0)
        saisie["BedroomAbvGr"] = st.selectbox(
            "Chambres", list(range(0, 9)), 3)
        saisie["TotRmsAbvGrd"] = st.slider("Pieces au total", 2, 15, 6)
        saisie["GarageCars"] = st.selectbox("Places de garage", [0, 1, 2, 3, 4], 2)
        saisie["GarageArea"] = st.number_input(
            "Surface du garage (pi2)", 0, 1500, 480, step=20)
        saisie["Fireplaces"] = st.selectbox("Cheminees", [0, 1, 2, 3], 0)

with onglet_detail:
    d1, d2 = st.columns(2)
    with d1:
        for champ, libelle in [("ExterQual", "Qualite du revetement exterieur"),
                               ("KitchenQual", "Qualite de la cuisine"),
                               ("BsmtQual", "Qualite du sous-sol"),
                               ("HeatingQC", "Qualite du chauffage")]:
            options = config["modalites"][champ]
            defaut = predicteur.defauts[champ]
            saisie[champ] = st.selectbox(
                libelle, options,
                index=options.index(defaut) if defaut in options else 0)
    with d2:
        for champ, libelle in [("CentralAir", "Climatisation centrale"),
                               ("GarageFinish", "Finition du garage"),
                               ("SaleCondition", "Condition de vente"),
                               ("HouseStyle", "Style de la maison")]:
            options = config["modalites"][champ]
            defaut = predicteur.defauts[champ]
            saisie[champ] = st.selectbox(
                libelle, options,
                index=options.index(defaut) if defaut in options else 0)
    saisie["YrSold"] = st.slider("Annee de vente", 2006, 2010, 2008)

st.divider()

if st.button("Estimer le prix", type="primary", use_container_width=True):
    with st.spinner("Calcul en cours..."):
        prix = predicteur.predire(saisie)
    marge = config["mape_validation"] / 100

    g1, g2, g3 = st.columns(3)
    g1.metric("Prix estime", f"{prix:,.0f} $".replace(",", " "))
    g2.metric("Fourchette basse", f"{prix * (1 - marge):,.0f} $".replace(",", " "))
    g3.metric("Fourchette haute", f"{prix * (1 + marge):,.0f} $".replace(",", " "))

    st.caption(
        f"La fourchette correspond a l'erreur relative moyenne du modele sur le "
        f"jeu de validation ({config['mape_validation']:.1f} %). Elle n'est pas un "
        f"intervalle de confiance statistique : environ la moitie des estimations "
        f"tombent en dehors.")

    with st.expander("Limites de cette estimation"):
        st.markdown(f"""
Le modele a ete entraine sur **{config['n_entrainement']} ventes reelles a Ames
(Iowa) entre 2006 et 2010**. Trois consequences a connaitre :

- **Il n'est pas transposable a un autre marche.** Les determinants du prix et
  leur poids different d'une ville a l'autre.
- **Les montants sont ceux de 2006-2010.** L'inflation immobiliere depuis n'est
  pas prise en compte.
- **Il predit mal les ventes atypiques** (construction inachevee, saisie, vente
  entre proches), car l'information qui explique leur prix n'est pas dans les
  donnees.

Metriques mesurees sur {config['n_validation']} maisons jamais vues par le modele :
RMSE {config['rmse_validation']:,.0f} \\$, erreur relative moyenne
{config['mape_validation']:.1f} %, R2 {config['r2_validation']:.3f}.
""")

with st.sidebar:
    st.header("A propos")
    st.markdown(f"""
**Pipeline complet**

1. Saisie completee par les valeurs par defaut du jeu d'entrainement
2. Nettoyage (etape 3 du notebook)
3. Creation des 13 variables derivees et encodage ordinal (etape 4)
4. Prediction par {config['nom_modele']} (etape 5)
5. Reconversion de log(prix) vers dollars

**Performance en validation**

| Metrique | Valeur |
|---|---|
| RMSE | {config['rmse_validation']:,.0f} \\$ |
| MAE | {config['mae_validation']:,.0f} \\$ |
| R2 | {config['r2_validation']:.3f} |
| Erreur relative moyenne | {config['mape_validation']:.1f} % |

Projet Fondements du Big Data, d'apres le notebook Kaggle
`gusthema/house-prices-prediction-using-tfdf`.
""")

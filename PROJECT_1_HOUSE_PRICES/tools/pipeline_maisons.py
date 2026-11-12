"""Classes du pipeline de preparation des donnees.

Elles sont definies ici, et non dans le notebook, pour une raison technique
importante : joblib serialise les objets par REFERENCE a leur module. Une classe
definie dans le notebook serait enregistree comme `__main__.NettoyeurMaisons`,
introuvable depuis l'application Streamlit. En les placant dans un module
importable, les artefacts sauvegardes sont rechargeables partout.

Le notebook affiche le code source de ces classes au moment ou il les utilise,
il n'y a donc pas de code cache.
"""
import numpy as np
import pandas as pd


class NettoyeurMaisons:
    """Traite les valeurs manquantes selon le contrat fit / transform.

    fit       : apprend les statistiques d'imputation (TRAIN uniquement).
    transform : applique ces statistiques (train, validation, production).
    """

    # Groupe A : "NA" = equipement absent (colonnes categorielles)
    ABSENCE_CATEGORIELLE = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    ]
    # Groupe B : equipement absent (colonnes numeriques)
    ABSENCE_NUMERIQUE = ["MasVnrArea"]
    # Groupe D : imputation par le mode
    MODE = ["Electrical", "MasVnrType"]
    # Colonnes retirees (justification en 3.4)
    A_RETIRER = ["Id", "Utilities"]

    def fit(self, df):
        self.lotfrontage_quartier_ = df.groupby("Neighborhood")["LotFrontage"].median()
        self.lotfrontage_globale_ = df["LotFrontage"].median()
        self.modes_ = {c: df[c].mode(dropna=True)[0] for c in self.MODE if c in df}
        self.colonnes_attendues_ = None
        self.colonnes_attendues_ = list(self.transform(df, _controle=False).columns)
        return self

    def transform(self, df, _controle=True):
        d = df.copy()

        for col in self.ABSENCE_CATEGORIELLE:
            if col in d:
                d[col] = d[col].fillna("Absent")

        for col in self.ABSENCE_NUMERIQUE:
            if col in d:
                d[col] = d[col].fillna(0)

        # Pas de garage : on utilise l'annee de construction de la maison.
        # Mettre 0 creerait une annee aberrante, a 2000 unites du reste.
        if "GarageYrBlt" in d:
            d["GarageYrBlt"] = d["GarageYrBlt"].fillna(d["YearBuilt"])

        if "LotFrontage" in d:
            par_quartier = d["Neighborhood"].map(self.lotfrontage_quartier_)
            d["LotFrontage"] = (d["LotFrontage"]
                                .fillna(par_quartier)
                                .fillna(self.lotfrontage_globale_))

        for col, valeur in self.modes_.items():
            if col in d:
                d[col] = d[col].fillna(valeur)

        d = d.drop(columns=[c for c in self.A_RETIRER if c in d])

        if _controle and self.colonnes_attendues_ is not None:
            attendues = [c for c in self.colonnes_attendues_ if c != "SalePrice"]
            manquantes = set(attendues) - set(d.columns)
            assert not manquantes, f"Colonnes manquantes : {manquantes}"
        return d

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class ConstructeurVariables:
    """Cree les variables derivees justifiees par l'analyse exploratoire."""

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        d = df.copy()

        # 1. Agregats de surface (constat : 5 colonnes redondantes)
        d["SurfaceTotale"] = d["TotalBsmtSF"] + d["1stFlrSF"] + d["2ndFlrSF"]
        d["SurfaceExterieure"] = (d["WoodDeckSF"] + d["OpenPorchSF"]
                                  + d["EnclosedPorch"] + d["3SsnPorch"]
                                  + d["ScreenPorch"])
        # Une demi-salle de bain vaut environ la moitie d'une complete
        d["TotalSdB"] = (d["FullBath"] + 0.5 * d["HalfBath"]
                         + d["BsmtFullBath"] + 0.5 * d["BsmtHalfBath"])

        # 2. Variables temporelles : age AU MOMENT DE LA VENTE, pas age
        # aujourd'hui (sinon le modele dependrait de la date d'execution)
        d["AgeVente"] = d["YrSold"] - d["YearBuilt"]
        d["AgeRenovation"] = d["YrSold"] - d["YearRemodAdd"]
        d["EstRenovee"] = (d["YearRemodAdd"] != d["YearBuilt"]).astype(int)
        d["EstNeuve"] = (d["AgeVente"] <= 1).astype(int)

        # 3. Indicateurs de presence (constat : inflation de zeros)
        d["ASousSol"] = (d["TotalBsmtSF"] > 0).astype(int)
        d["AGarage"] = (d["GarageArea"] > 0).astype(int)
        d["ACheminee"] = (d["Fireplaces"] > 0).astype(int)
        d["APiscine"] = (d["PoolArea"] > 0).astype(int)
        d["AEtage"] = (d["2ndFlrSF"] > 0).astype(int)

        # 4. Interaction qualite x surface (divisee par 1000 pour la lisibilite)
        d["QualiteFoisSurface"] = d["OverallQual"] * d["GrLivArea"] / 1000

        return d

    def fit_transform(self, df, y=None):
        return self.fit(df).transform(df)

    @staticmethod
    def nouvelles():
        return ["SurfaceTotale", "SurfaceExterieure", "TotalSdB",
                "AgeVente", "AgeRenovation", "EstRenovee", "EstNeuve",
                "ASousSol", "AGarage", "ACheminee", "APiscine", "AEtage",
                "QualiteFoisSurface"]


class PreparationArbres:
    """Variables creees et encodage ordinal. Nominales laissees en texte."""

    def __init__(self, echelles):
        self.echelles = echelles
        self.constructeur = ConstructeurVariables()

    def fit(self, df, y=None):
        self.colonnes_ = None
        self.colonnes_ = list(self.transform(df).columns)
        return self

    def transform(self, df):
        d = self.constructeur.transform(df)
        for col, echelle in self.echelles.items():
            if col in d:
                # Modalite inconnue -> minimum de l'echelle (choix conservateur :
                # on ne suppose pas une qualite elevee)
                d[col] = d[col].map(echelle).fillna(min(echelle.values())).astype(int)
        # MSSubClass est un code numerique mais c'est une categorie (cf. etape 1)
        if "MSSubClass" in d:
            d["MSSubClass"] = d["MSSubClass"].astype(str)
        return d.drop(columns=["SalePrice"], errors="ignore")

    def fit_transform(self, df, y=None):
        return self.fit(df).transform(df)


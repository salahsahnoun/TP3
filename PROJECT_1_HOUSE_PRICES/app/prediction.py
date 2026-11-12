"""Chaine de prediction pour le deploiement.

Ce module est le SEUL endroit ou la logique de prediction est ecrite. Le
notebook et l'application Streamlit l'importent tous les deux, ce qui garantit
qu'une maison saisie dans l'interface subit exactement le meme traitement
qu'une maison du jeu d'entrainement.

La chaine est : saisie partielle -> completion par les valeurs par defaut
-> nettoyeur (etape 3) -> preparation arbres (etape 4) -> modele (etape 5).
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
MODELS = RACINE / "models"


class Predicteur:
    """Charge les artefacts une fois, puis predit a la demande."""

    def __init__(self, dossier_modeles=MODELS):
        self.dossier = Path(dossier_modeles)
        self.config = json.loads((self.dossier / "deploiement.json").read_text())
        self.defauts = self.config["valeurs_defaut"]
        self.colonnes_brutes = self.config["colonnes_brutes"]
        self.dtypes_bruts = self.config["dtypes_bruts"]
        self.colonnes_preparees = self.config["colonnes_preparees"]
        self.dtypes_prepares = self.config["dtypes_prepares"]

        self.nettoyeur = joblib.load(self.dossier / "nettoyeur.joblib")
        self.preparation = joblib.load(self.dossier / "preprocessor.joblib")["prep_arbres"]

        # Import tardif : TensorFlow met plusieurs secondes a se charger, on
        # evite de le payer si le module est seulement importe pour inspection.
        import tensorflow_decision_forests as tfdf
        import tf_keras
        self._tfdf = tfdf
        self.modele = tf_keras.models.load_model(
            str(self.dossier / self.config["modele"]))

    def completer(self, saisie):
        """Complete une saisie partielle avec les valeurs par defaut du train.

        Les defauts sont la mediane (numerique) ou le mode (categoriel) calcules
        sur le jeu d'ENTRAINEMENT uniquement : aucune information du jeu de
        validation n'entre dans l'application.
        """
        ligne = dict(self.defauts)
        inconnues = set(saisie) - set(self.colonnes_brutes)
        if inconnues:
            raise ValueError(f"Colonnes inconnues : {sorted(inconnues)}")
        ligne.update({k: v for k, v in saisie.items() if v is not None})
        brut = pd.DataFrame([ligne])[self.colonnes_brutes]

        # Les valeurs par defaut sont lues depuis du JSON, donc toutes en float
        # ou en str. On restaure les types du jeu d'entrainement, sans quoi
        # MSSubClass deviendrait la categorie "50.0" au lieu de "50".
        for col, dtype in self.dtypes_bruts.items():
            if col in brut and not brut[col].isna().any():
                try:
                    brut[col] = brut[col].astype(dtype)
                except (ValueError, TypeError):
                    pass
        return brut

    def _aligner(self, prepare):
        """Aligne colonnes et types sur ceux vus a l'entrainement.

        TFDF enregistre une signature typee : une colonne passee en float64
        alors que le modele attend un int64 provoque une erreur au moment de
        la prediction. Ce realignement est donc obligatoire, pas cosmetique.
        """
        prepare = prepare[self.colonnes_preparees].copy()
        for col, dtype in self.dtypes_prepares.items():
            if str(prepare[col].dtype) != dtype:
                prepare[col] = prepare[col].astype(dtype)
        return prepare

    def predire(self, saisie):
        """Retourne le prix estime en dollars pour une maison."""
        brut = self.completer(saisie)
        propre = self.nettoyeur.transform(brut)
        prepare = self._aligner(self.preparation.transform(propre))
        jeu = self._tfdf.keras.pd_dataframe_to_tf_dataset(prepare)
        # Le modele a ete entraine sur log1p(prix) : expm1 fait l'inverse
        return float(np.expm1(self.modele.predict(jeu, verbose=0).ravel()[0]))

    def predire_lot(self, dataframe_brut):
        """Meme chose pour un DataFrame de maisons brutes aux colonnes completes."""
        propre = self.nettoyeur.transform(dataframe_brut)
        prepare = self._aligner(self.preparation.transform(propre))
        jeu = self._tfdf.keras.pd_dataframe_to_tf_dataset(prepare)
        return np.expm1(self.modele.predict(jeu, verbose=0).ravel())


_PREDICTEUR = None


def obtenir_predicteur():
    """Singleton : evite de recharger TensorFlow a chaque appel."""
    global _PREDICTEUR
    if _PREDICTEUR is None:
        _PREDICTEUR = Predicteur()
    return _PREDICTEUR

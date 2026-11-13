"""API d'estimation de la duree d'un trajet en taxi a New York.

Lancement :
    uvicorn app.api:api --host 127.0.0.1 --port 8000

Toute la logique vit dans app/prediction_spark.py, qui est aussi le module
teste dans le notebook : l'API ne duplique aucun calcul.
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction_spark import obtenir_predicteur

RACINE = Path(__file__).resolve().parent.parent
CONFIG = json.loads((RACINE / "models" / "config.json").read_text())

api = FastAPI(
    title="Duree de trajet en taxi, New York",
    description=(
        "Estime la duree d'une course a partir des seules informations connues "
        "au moment de la prise en charge. "
        f"Modele : {CONFIG['modele']}, entraine sur {CONFIG['n_entrainement']:,} courses "
        f"de janvier 2016.".replace(",", " ")),
    version="1.0",
)


class Course(BaseModel):
    pickup_datetime: str = Field(..., examples=["2016-01-28 18:30:00"],
                                 description="Horodatage de la prise en charge")
    PULocationID: int = Field(..., ge=1, le=263, description="Zone TLC de depart")
    DOLocationID: int = Field(..., ge=1, le=263, description="Zone TLC d'arrivee")
    passenger_count: int = Field(1, ge=1, le=6)
    RatecodeID: int = Field(1, ge=1, le=6)
    VendorID: int = Field(1, ge=1, le=2)
    store_and_fwd_flag: str = Field("N", pattern="^[YN]$")


@api.get("/")
def racine():
    return {
        "service": "estimation de duree de trajet",
        "modele": CONFIG["modele"],
        "entraine_sur": CONFIG["n_entrainement"],
        "rmse_test_s": CONFIG["rmse_test_s"],
        "mape_test_pct": CONFIG["mape_test_pct"],
        "documentation": "/docs",
    }


@api.get("/sante")
def sante():
    return {"etat": "ok", "spark": obtenir_predicteur().spark.version}


@api.post("/predire")
def predire(course: Course):
    try:
        secondes = obtenir_predicteur().predire([course.model_dump()])[0]
    except Exception as erreur:
        raise HTTPException(status_code=500, detail=str(erreur))
    return {
        "duree_estimee_s": round(secondes),
        "duree_estimee_min": round(secondes / 60, 1),
        "marge_relative_pct": CONFIG["mape_test_pct"],
        "avertissement": (
            "Modele entraine sur janvier 2016 uniquement. Ni meteo ni trafic "
            "temps reel ne sont pris en compte."),
    }

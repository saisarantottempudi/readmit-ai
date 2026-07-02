"""Request/response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class PatientEncounter(BaseModel):
    race: str = "missing"
    gender: Literal["Male", "Female"]
    age: str = Field(examples=["[70-80)"])
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=30)
    admission_source_id: int = Field(ge=1, le=26)
    diag_group: str = Field(examples=["circulatory", "diabetes"])
    max_glu_serum: str = "missing"
    A1Cresult: str = "missing"
    insulin: str = "No"
    change: str = "No"
    diabetesMed: Literal["Yes", "No"]
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    number_diagnoses: int = Field(ge=1)


class Factor(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    risk_score: float
    risk_band: Literal["low", "medium", "high"]
    top_factors: list[Factor]
    model_version: str


def risk_band(score: float) -> Literal["low", "medium", "high"]:
    if score < 0.3:
        return "low"
    if score < 0.6:
        return "medium"
    return "high"

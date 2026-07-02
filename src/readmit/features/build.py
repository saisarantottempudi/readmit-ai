"""Feature definitions and preprocessing shared by training and serving."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_group",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    """One-hot categoricals (unknowns ignored at serve time), scale numerics."""
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

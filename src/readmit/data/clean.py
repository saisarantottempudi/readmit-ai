"""Cleaning and target construction for the readmission dataset."""

import numpy as np
import pandas as pd

TARGET = "readmitted_30d"

# Discharge dispositions indicating death or hospice — readmission impossible (label leakage).
LEAKAGE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

DROP_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "weight",          # ~97% missing
    "payer_code",      # ~40% missing, not clinical
    "medical_specialty",  # ~50% missing
    "diag_2",
    "diag_3",
    "readmitted",
]

# ICD-9 chapter grouping for primary diagnosis.
_ICD9_CHAPTERS = [
    (139, "infectious"),
    (239, "neoplasms"),
    (279, "endocrine"),
    (289, "blood"),
    (319, "mental"),
    (389, "nervous"),
    (459, "circulatory"),
    (519, "respiratory"),
    (579, "digestive"),
    (629, "genitourinary"),
    (679, "pregnancy"),
    (709, "skin"),
    (739, "musculoskeletal"),
    (759, "congenital"),
    (779, "perinatal"),
    (799, "ill_defined"),
    (999, "injury"),
]


def icd9_group(code: str | float | None) -> str:
    """Map an ICD-9 code to its chapter group. V/E codes and missing → 'other'."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "other"
    text = str(code)
    if text.startswith(("V", "E")):
        return "other"
    try:
        value = float(text)
    except ValueError:
        return "other"
    if 250 <= value < 251:
        return "diabetes"
    for upper, name in _ICD9_CHAPTERS:
        if value <= upper:
            return name
    return "other"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Build the modelling frame: target, leakage removal, missing markers, diag grouping."""
    out = df.replace("?", np.nan).infer_objects(copy=False)

    out[TARGET] = (out["readmitted"] == "<30").astype(int)
    out = out[~out["discharge_disposition_id"].isin(LEAKAGE_DISPOSITIONS)]
    out = out[out["gender"] != "Unknown/Invalid"]

    out["diag_group"] = out["diag_1"].map(icd9_group)
    out = out.drop(columns=[c for c in DROP_COLUMNS + ["diag_1"] if c in out.columns])

    for col in ("race", "max_glu_serum", "A1Cresult"):
        if col in out.columns:
            out[col] = out[col].replace("None", np.nan).fillna("missing")

    return out.reset_index(drop=True)

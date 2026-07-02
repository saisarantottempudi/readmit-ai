"""Pandera schema validation for the raw UCI diabetes dataset."""

import pandas as pd
import pandera as pa

AGE_BANDS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]

RawSchema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(int, unique=True),
        "patient_nbr": pa.Column(int),
        "race": pa.Column(str, nullable=True),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female", "Unknown/Invalid"])),
        "age": pa.Column(str, pa.Check.isin(AGE_BANDS)),
        "admission_type_id": pa.Column(int, pa.Check.in_range(1, 8)),
        "discharge_disposition_id": pa.Column(int, pa.Check.in_range(1, 30)),
        "admission_source_id": pa.Column(int, pa.Check.in_range(1, 26)),
        "time_in_hospital": pa.Column(int, pa.Check.in_range(1, 14)),
        "num_lab_procedures": pa.Column(int, pa.Check.ge(0)),
        "num_procedures": pa.Column(int, pa.Check.ge(0)),
        "num_medications": pa.Column(int, pa.Check.ge(0)),
        "number_outpatient": pa.Column(int, pa.Check.ge(0)),
        "number_emergency": pa.Column(int, pa.Check.ge(0)),
        "number_inpatient": pa.Column(int, pa.Check.ge(0)),
        "diag_1": pa.Column(str, nullable=True),
        "number_diagnoses": pa.Column(int, pa.Check.ge(1)),
        "max_glu_serum": pa.Column(str, nullable=True),
        "A1Cresult": pa.Column(str, nullable=True),
        "insulin": pa.Column(str),
        "change": pa.Column(str),
        "diabetesMed": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "readmitted": pa.Column(str, pa.Check.isin(["<30", ">30", "NO"])),
    },
    strict=False,  # raw file has extra medication columns we ignore
    coerce=True,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw dataframe; raises pandera.errors.SchemaError with details on failure."""
    return RawSchema.validate(df, lazy=True)

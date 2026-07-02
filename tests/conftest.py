import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Small synthetic frame matching the raw UCI schema."""
    n = 200
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": rng.integers(1, 10_000, n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", "?"], n),
            "gender": rng.choice(["Male", "Female"], n),
            "age": rng.choice(["[50-60)", "[60-70)", "[70-80)"], n),
            "weight": ["?"] * n,
            "admission_type_id": rng.integers(1, 9, n),
            "discharge_disposition_id": rng.choice([1, 2, 3, 6, 11], n),
            "admission_source_id": rng.integers(1, 10, n),
            "time_in_hospital": rng.integers(1, 15, n),
            "payer_code": ["?"] * n,
            "medical_specialty": ["?"] * n,
            "num_lab_procedures": rng.integers(0, 100, n),
            "num_procedures": rng.integers(0, 7, n),
            "num_medications": rng.integers(1, 40, n),
            "number_outpatient": rng.integers(0, 5, n),
            "number_emergency": rng.integers(0, 5, n),
            "number_inpatient": rng.integers(0, 5, n),
            "diag_1": rng.choice(["250.01", "428", "V57", "780", "?"], n),
            "diag_2": ["?"] * n,
            "diag_3": ["?"] * n,
            "number_diagnoses": rng.integers(1, 10, n),
            "max_glu_serum": rng.choice(["None", ">200", "Norm"], n),
            "A1Cresult": rng.choice(["None", ">7", "Norm"], n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
            "readmitted": rng.choice(["<30", ">30", "NO"], n, p=[0.15, 0.35, 0.5]),
        }
    )

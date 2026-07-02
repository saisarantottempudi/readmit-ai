import numpy as np
import pandas as pd
import pandera as pa
import pytest

from readmit.data.clean import LEAKAGE_DISPOSITIONS, TARGET, clean, icd9_group
from readmit.data.schema import validate_raw
from readmit.data.split import make_splits


class TestSchema:
    def test_valid_frame_passes(self, raw_frame):
        validated = validate_raw(raw_frame)
        assert len(validated) == len(raw_frame)

    def test_bad_readmitted_value_fails(self, raw_frame):
        bad = raw_frame.copy()
        bad.loc[0, "readmitted"] = "MAYBE"
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_duplicate_encounter_id_fails(self, raw_frame):
        bad = raw_frame.copy()
        bad.loc[1, "encounter_id"] = bad.loc[0, "encounter_id"]
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)


class TestClean:
    def test_target_built_from_readmitted(self, raw_frame):
        cleaned = clean(raw_frame)
        kept = raw_frame[~raw_frame["discharge_disposition_id"].isin(LEAKAGE_DISPOSITIONS)]
        expected_rate = (kept["readmitted"] == "<30").mean()
        assert cleaned[TARGET].mean() == pytest.approx(expected_rate)

    def test_leakage_dispositions_removed(self, raw_frame):
        cleaned = clean(raw_frame)
        assert not set(cleaned["discharge_disposition_id"]) & LEAKAGE_DISPOSITIONS

    def test_id_and_leaky_columns_dropped(self, raw_frame):
        cleaned = clean(raw_frame)
        for col in ("encounter_id", "patient_nbr", "readmitted", "weight", "diag_1"):
            assert col not in cleaned.columns

    def test_question_marks_become_missing_category(self, raw_frame):
        cleaned = clean(raw_frame)
        assert "?" not in set(cleaned["race"])
        assert "missing" in set(cleaned["race"])


class TestIcd9Group:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("250.01", "diabetes"),
            ("428", "circulatory"),
            ("486", "respiratory"),
            ("V57", "other"),
            ("E812", "other"),
            (np.nan, "other"),
        ],
    )
    def test_grouping(self, code, expected):
        assert icd9_group(code) == expected


class TestSplits:
    def test_split_names_and_sizes(self, raw_frame):
        cleaned = clean(raw_frame)
        splits = make_splits(cleaned, reference_size=50)
        assert set(splits) == {"train", "test", "reference"}
        assert len(splits["train"]) + len(splits["test"]) == len(cleaned)
        assert len(splits["reference"]) == 50

    def test_split_is_deterministic(self, raw_frame):
        cleaned = clean(raw_frame)
        first = make_splits(cleaned, reference_size=50)
        second = make_splits(cleaned, reference_size=50)
        pd.testing.assert_frame_equal(first["train"], second["train"])

    def test_stratification_preserves_rate(self, raw_frame):
        cleaned = clean(raw_frame)
        splits = make_splits(cleaned, reference_size=50)
        overall = cleaned[TARGET].mean()
        assert splits["test"][TARGET].mean() == pytest.approx(overall, abs=0.05)

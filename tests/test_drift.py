import pytest

from readmit.data.clean import clean
from readmit.monitoring.drift import compute_drift


@pytest.fixture
def reference(raw_frame):
    return clean(raw_frame)


class TestComputeDrift:
    def test_no_drift_on_identical_data(self, reference):
        result = compute_drift(reference, reference.copy())
        assert result.share_drifted == 0.0
        assert result.drifted_columns == []

    def test_detects_shifted_numeric_features(self, reference):
        shifted = reference.copy()
        shifted["num_medications"] = shifted["num_medications"] * 2 + 20
        shifted["time_in_hospital"] = (shifted["time_in_hospital"] + 6).clip(upper=14)
        result = compute_drift(reference, shifted)
        assert result.share_drifted > 0.0
        assert "num_medications" in result.drifted_columns

    def test_report_html_generated(self, reference):
        result = compute_drift(reference, reference.copy())
        assert "<html" in result.report_html.lower()

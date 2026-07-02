import pandas as pd
import pytest

from readmit.data.clean import clean
from readmit.features.build import FEATURES
from readmit.train.evaluate import evaluate
from readmit.train.pipeline import imbalance_weight, make_model, prepare_xy
from readmit.train.promote import should_promote


class TestPromotion:
    def test_promotes_when_no_champion(self):
        assert should_promote(0.5, None) is True

    def test_promotes_when_strictly_better(self):
        assert should_promote(0.66, 0.65) is True

    def test_keeps_champion_on_tie(self):
        assert should_promote(0.65, 0.65) is False

    def test_keeps_champion_when_worse(self):
        assert should_promote(0.60, 0.65) is False


class TestFeatures:
    def test_features_exist_in_cleaned_frame(self, raw_frame):
        cleaned = clean(raw_frame)
        missing = [f for f in FEATURES if f not in cleaned.columns]
        assert missing == []

    def test_prepare_xy_shapes_and_types(self, raw_frame):
        cleaned = clean(raw_frame)
        X, y = prepare_xy(cleaned)
        assert list(X.columns) == FEATURES
        assert len(X) == len(y)
        assert X["admission_type_id"].dtype == object  # ids treated as categories


class TestPipeline:
    @pytest.fixture
    def trained(self, raw_frame):
        cleaned = clean(raw_frame)
        X, y = prepare_xy(cleaned)
        model = make_model(scale_pos_weight=imbalance_weight(y))
        model.fit(X, y)
        return model, X, y

    def test_predict_proba_in_unit_interval(self, trained):
        model, X, _ = trained
        proba = model.predict_proba(X)[:, 1]
        assert proba.min() >= 0.0 and proba.max() <= 1.0

    def test_evaluate_returns_expected_metrics(self, trained):
        model, X, y = trained
        metrics = evaluate(model, X, y)
        assert set(metrics) == {"auc", "average_precision", "brier"}
        assert 0.5 <= metrics["auc"] <= 1.0

    def test_handles_unseen_category_at_predict_time(self, trained):
        model, X, _ = trained
        row = X.head(1).copy()
        row["race"] = "NeverSeenBefore"
        proba = model.predict_proba(row)
        assert proba.shape == (1, 2)


class TestImbalanceWeight:
    def test_weight_matches_class_ratio(self):
        y = pd.Series([1, 0, 0, 0])
        assert imbalance_weight(y) == 3.0

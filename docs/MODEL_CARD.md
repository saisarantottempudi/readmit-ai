# Model Card — readmit-ai

## Model details

- **Task**: Binary classification — probability of unplanned hospital readmission within 30 days of discharge.
- **Architecture**: scikit-learn preprocessing pipeline (one-hot categoricals, scaled numerics) + XGBoost gradient-boosted trees with class-imbalance weighting.
- **Registry**: MLflow model `readmit`, production alias assigned by champion/challenger gate (challenger must strictly beat the champion's held-out AUC).
- **Version**: see `/health` endpoint (`model_version`) or the MLflow registry.

## Intended use

- Decision **support** for discharge planning and post-discharge follow-up prioritisation.
- Ranking patients for review by a clinical team; the risk band (`low` / `medium` / `high`) is a triage aid.
- **Not** for automated denial of care, insurance pricing, or any fully automated clinical decision. A clinician must remain in the loop.

## Training data

- UCI **Diabetes 130-US hospitals** dataset (1999–2008), 101,766 inpatient encounters of diabetic patients across 130 US hospitals.
- After cleaning: 99,340 encounters; positive class (readmitted <30 days) ≈ 11.4%.
- Encounters ending in death or hospice discharge are removed (readmission is undefined — label leakage).

## Performance

| Metric | Held-out test |
|--------|---------------|
| AUC | 0.674 |
| Positive rate | 0.114 |

Comparable published baselines on this dataset report AUC ≈ 0.65–0.69. Top risk drivers (global SHAP): prior inpatient admissions, prior emergency visits, discharge disposition, number of medications.

## Explainability

Every prediction returns the top-3 signed SHAP contributions, so a reviewer can see *why* a patient scored high (e.g. "3 prior inpatient stays"). Global SHAP importance is logged as an MLflow artifact per training run.

## Limitations & biases

- **Population shift**: data is US, diabetic-only, 1999–2008. Recalibration and validation on the local population (e.g. an NHS trust) is mandatory before any clinical use.
- **Race field**: recorded inconsistently in the source data (~2% missing, coded as `missing`). Model predictions should be audited for differential performance across race and gender groups before deployment; this repository ships the tooling (SHAP, drift reports) but not a completed fairness audit.
- **Label definition**: "readmission" is any inpatient return within 30 days at a participating hospital — readmissions elsewhere are missed (label noise).
- **Moderate discrimination**: AUC 0.67 means the model is a prioritisation aid, not a diagnostic instrument.

## Monitoring & retraining

- Evidently compares live traffic against a frozen reference split; drift share above threshold (default 0.3) triggers automated retraining.
- Retrained challengers are only promoted if they beat the champion on held-out AUC — the system never silently degrades.

## Governance notes (NHS context)

Aligned with the spirit of the NHS AI assurance guidance: human-in-the-loop, per-prediction explainability, drift surveillance, versioned and auditable model registry, and a documented promotion gate. A DPIA, clinical safety case (DCB0129/DCB0160), and local validation would be required before real-world deployment.

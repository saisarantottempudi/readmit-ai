"""Send simulated (optionally drifted) traffic to the prediction API.

Usage:
    python scripts/simulate_traffic.py --n 200
    python scripts/simulate_traffic.py --n 200 --drift
"""

import argparse
import sys

import pandas as pd
import requests

from readmit.config import PROCESSED_DIR
from readmit.features.build import FEATURES


def build_payloads(n: int, drift: bool) -> list[dict]:
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    sample = test.sample(n=min(n, len(test)), random_state=1)[FEATURES].copy()

    if drift:
        # Simulate an older, sicker population arriving.
        sample["num_medications"] = (sample["num_medications"] * 2 + 10).clip(upper=80)
        sample["time_in_hospital"] = (sample["time_in_hospital"] + 5).clip(upper=14)
        sample["number_inpatient"] = sample["number_inpatient"] + 3
        sample["age"] = "[80-90)"

    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        sample[col] = sample[col].astype(int)
    return sample.to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()

    payloads = build_payloads(args.n, args.drift)
    ok = failed = 0
    for payload in payloads:
        response = requests.post(f"{args.url}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            ok += 1
        else:
            failed += 1
            if failed <= 3:
                print(f"FAIL {response.status_code}: {response.text[:200]}")

    print(f"sent={len(payloads)} ok={ok} failed={failed} drift={args.drift}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Curated showcase patients for the dashboard demo.

12 diverse synthetic patients (high/low cardiac risk, diabetes, respiratory
flag, medication interactions, conflicting signals) with images included.
Deterministic: same seed -> same showcase.
"""
from data.synth import generate_patients

_SHOWCASE = None


def get_showcase_patients(ctx=None, n=400, seed=7):
    """Pick 12 interesting patients from a generated pool."""
    global _SHOWCASE
    if _SHOWCASE is not None:
        return _SHOWCASE
    pool = generate_patients(n, seed=seed, with_images=True)

    def score(p):
        # diversity score: prefer interesting cases
        s = 0
        labs = p["labels"]
        if labs["cardiac_risk"]:
            s += 2
        if labs["diabetes_risk"]:
            s += 2
        if labs["respiratory_flag"]:
            s += 2
        if len(p["medications"]) >= 2:
            s += 1
        return s

    ranked = sorted(pool, key=score, reverse=True)
    picked, seen_scores = [], set()
    # take top diverse: ensure at least 2 low-risk for contrast
    for p in ranked:
        if len(picked) >= 10:
            break
        picked.append(p)
    low = [p for p in pool if sum(p["labels"].values()) == 0][:2]
    picked.extend(low)
    for i, p in enumerate(picked):
        p["patient_id"] = f"SYN-SHOW-{i + 1:02d}"
    _SHOWCASE = picked
    return _SHOWCASE


def get_patient(ctx, patient_id):
    for p in get_showcase_patients(ctx):
        if p["patient_id"] == patient_id:
            return p
    return None

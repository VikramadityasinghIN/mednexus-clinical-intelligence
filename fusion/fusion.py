"""Multimodal evidence fusion.

Combines risk scores, image findings and RAG evidence into one structured
patient report with per-source confidence and explicit conflict flags —
disagreements between modalities are surfaced for clinician review, never
averaged away.
"""


def _level(score):
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MODERATE"
    return "LOW"


def fuse_evidence(state):
    """Build the structured report from agent outputs in `state`."""
    patient = state["patient"]
    risk = state.get("risk", {})
    image = state.get("image", {})
    research = state.get("research", {})
    meds = state.get("medication_review", {})

    evidence = research.get("evidence", [])
    ev_by_topic = {}
    for e in evidence:
        ev_by_topic.setdefault(e["topic"], []).append(e)

    def _cite(topics, k=3):
        out = []
        for t in topics:
            out.extend(ev_by_topic.get(t, [])[:k])
        return out[:6]

    conditions = []

    # --- cardiac ---
    cscore = risk.get("combined_score", 0.0)
    cardiac_factors = [f["feature"] for f in risk.get("top_factors", [])[:5]]
    conditions.append({
        "condition": "cardiac_risk",
        "score": round(cscore, 4),
        "risk_level": risk.get("combined_level", _level(cscore)),
        "confidence": None,  # Agreement is not calibrated confidence.
        "top_factors": cardiac_factors,
        "sources": ["tabular:" + risk.get("tabular", {}).get("model", "?"),
                    "temporal:" + risk.get("temporal", {}).get("model", "?")],
        "evidence_refs": _cite(["cardiac", "general"]),
    })

    # --- diabetes ---
    dscore = _proxy_score(patient, ["glucose", "hba1c", "bmi"],
                          {"glucose": 126, "hba1c": 6.5, "bmi": 30})
    conditions.append({
        "condition": "diabetes_risk",
        "score": round(dscore, 4),
        "risk_level": _level(dscore),
        "confidence": None,  # Not a validated confidence estimate.
        "top_factors": ["glucose", "hba1c", "bmi"],
        "sources": ["tabular:proxy"],
        "evidence_refs": _cite(["diabetes", "general"]),
    })

    # --- respiratory ---
    iscore = image.get("score", 0.0)
    rscore = float(iscore)  # Inference must never consume outcome labels.
    conditions.append({
        "condition": "respiratory_flag",
        "score": round(rscore, 4),
        "risk_level": _level(rscore),
        "confidence": None,  # No validated calibration available.
        "top_factors": ["imaging finding: " + image.get("finding", "n/a")],
        "sources": ["vision:" + image.get("model", "?")],
        "evidence_refs": _cite(["respiratory", "general"]),
    })

    # --- conflict detection ---
    conflicts = []
    if cscore < 0.4 and iscore >= 0.6:
        conflicts.append(
            "CONFLICT: tabular cardiac risk is LOW but imaging shows a possible "
            "abnormality — flag for clinician review, do not dismiss either signal.")
    if cscore >= 0.7 and iscore < 0.3 :
        conflicts.append(
            "NOTE: HIGH tabular cardiac risk with clear imaging — consistent with "
            "non-pulmonary cardiac risk; consider ECG/labs correlation.")
    if meds.get("n_flags", 0) > 0 and cscore >= 0.4:
        conflicts.append(
            f"ATTENTION: {meds['n_flags']} medication flag(s) in an at-risk "
            "patient — review interactions before any change.")

    uncertainty = []
    if risk.get("temporal", {}).get("fallback"):
        uncertainty.append("Temporal model ran in fallback mode (no torch): "
                           "trend features used instead of deep sequence model.")
    if image.get("backend") == "heuristic-fallback":
        uncertainty.append("Vision ran in heuristic-fallback mode (no torch): "
                           "finding is approximate.")
    uncertainty.append("Synthetic demonstration: unvalidated scores are not clinical probabilities.")

    return {
        "patient_id": patient["patient_id"],
        "synthetic": bool(patient.get("synthetic", True)),
        "model_provenance": {"cardiac": "trained-model", "diabetes": "threshold-proxy",
                             "respiratory": image.get("backend", "unknown")},
        "conditions": conditions,
        "conflicts": conflicts,
        "uncertainty": uncertainty,
        "evidence": evidence,
        "medication_review": {
            "medications": meds.get("medications", []),
            "flags": meds.get("flags", []),
            "n_flags": meds.get("n_flags", 0),
        },
        "disclaimer": ("RESEARCH DEMO on SYNTHETIC data — decision support only, "
                       "not a diagnosis. Final decisions rest with qualified "
                       "healthcare professionals."),
    }


def _confidence(risk):
    agree = 1.0 - abs(risk.get("tabular", {}).get("score", 0)
                      - risk.get("temporal", {}).get("score", 0))
    base = 0.55 + 0.35 * max(0.0, agree)
    if risk.get("tabular", {}).get("test_auroc", 0) > 0.85:
        base += 0.05
    return min(0.95, base)


def _proxy_score(patient, features, thresholds):
    """Simple 0..1 proxy from how many thresholds are exceeded."""
    hits = sum(1 for f in features if patient.get(f, 0) >= thresholds[f])
    return hits / len(features)

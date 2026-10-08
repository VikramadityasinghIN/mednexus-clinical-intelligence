"""Offline tests — everything testable without torch/ollama/gradio.

Heavy tests (agents/orchestrator) share one small SystemContext fixture.
Run:  python run_tests.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

TMP = tempfile.mkdtemp(prefix="ci_test_")
_ctx = None


def get_ctx():
    global _ctx
    if _ctx is None:
        from agents.context import SystemContext
        _ctx = SystemContext(n_patients=300, seed=11,
                             artifacts_dir=os.path.join(TMP, "artifacts"))
        _ctx.train_all(verbose=False)
    return _ctx


# ---------------- data ----------------
def test_synth_determinism():
    from data.synth import generate_patients
    a = generate_patients(20, seed=5)
    b = generate_patients(20, seed=5)
    assert a[0]["patient_id"] == b[0]["patient_id"]
    assert a[7]["total_chol"] == b[7]["total_chol"]
    assert np.array_equal(a[3]["image"], b[3]["image"])
    assert a[9]["note"] == b[9]["note"]


def test_synth_modalities():
    from data.synth import generate_patients
    ps = generate_patients(30, seed=6)
    for p in ps:
        assert 4 <= len(p["visits"]) <= 8
        assert len(p["note"]) > 50
        assert p["image"].shape == (224, 224)
        assert set(p["labels"]) == {"cardiac_risk", "diabetes_risk", "respiratory_flag"}
        assert p["synthetic"] is True


def test_synth_no_images_mode():
    from data.synth import generate_patients
    ps = generate_patients(10, seed=6, with_images=False)
    assert ps[0]["image"] is None
    assert ps[0]["image_label"] in (0, 1)


# ---------------- tabular ----------------
def _small_xy():
    from data.synth import TABULAR_FEATURES, generate_patients, patients_to_frame
    ps = generate_patients(200, seed=8, with_images=False)
    f = patients_to_frame(ps)
    return f[TABULAR_FEATURES], f["cardiac_risk"]


def test_tabular_roundtrip():
    import tempfile
    from models.tabular import MODEL_CLASSES
    X, y = _small_xy()
    for name, cls in MODEL_CLASSES.items():
        m = cls(seed=0).fit(X, y)
        p1 = m.predict_proba(X)
        assert p1.shape == (len(X),) and np.all((p1 >= 0) & (p1 <= 1))
        path = os.path.join(tempfile.mkdtemp(), f"{name}.pkl")
        m.save(path)
        m2 = cls.load(path)
        assert np.allclose(p1, m2.predict_proba(X))
        assert len(m.feature_importance()) > 0


def test_compare_models():
    from models.tabular import compare_models
    X, y = _small_xy()
    res = compare_models(X, y, X, y, seed=0)
    assert set(res) == {"xgboost", "random_forest", "logistic_regression"}
    for v in res.values():
        assert 0.5 <= v["metrics"]["auroc"] <= 1.0  # train==test here


# ---------------- temporal ----------------
def test_temporal_fallback():
    from data.synth import generate_patients, patients_to_sequences
    from models.temporal import (TrendFeatureModel, available_temporal_models,
                                 trend_features)
    assert "trend_fallback" in available_temporal_models()
    ps = generate_patients(60, seed=9, with_images=False)
    X, mask = patients_to_sequences(ps)
    y = np.array([p["labels"]["cardiac_risk"] for p in ps])
    F = trend_features(X, mask)
    assert F.shape == (60, 25)  # 5 channels x 5 stats
    m = TrendFeatureModel(seed=0).fit(X, mask, y)
    p = m.predict_proba(X, mask)
    assert p.shape == (60,)


# ---------------- nlp ----------------
def test_nlp_extraction():
    from models.nlp import extract_findings, extract_vitals
    note = ("62-year-old male presents with chest discomfort on exertion. "
            "Denies chest pain at rest. BP 150/95.")
    res = extract_findings(note)
    kinds = {(e["finding"], e["negated"]) for e in res["rule_based"]}
    assert ("chest_pain", False) in kinds
    assert ("chest_pain", True) in kinds  # negated mention caught
    assert extract_vitals(note)["bp"] == (150, 95)


# ---------------- vision ----------------
def test_vision_fallback():
    from data.synth import generate_synthetic_xray
    from models import vision as V
    rng = np.random.RandomState(3)
    ab = np.array([generate_synthetic_xray(rng, abnormal=True)[0] for _ in range(10)])
    no = np.array([generate_synthetic_xray(rng, abnormal=False)[0] for _ in range(10)])
    vm = V.get_vision_model()
    sa, sn = vm.predict_proba(ab), vm.predict_proba(no)
    assert sa.mean() > sn.mean(), "heuristic should separate synthetic classes"
    hm = vm.heatmap(ab[0])
    assert np.asarray(hm).shape == (224, 224)


# ---------------- rag ----------------
def test_rag_citations():
    from rag.retriever import EvidenceRetriever
    r = EvidenceRetriever()
    hits = r.retrieve("high LDL cholesterol heart risk", k=3)
    assert len(hits) == 3
    for h in hits:
        assert {"id", "title", "text", "topic", "source", "score"} <= set(h)
        assert h["source"]
    assert r.precision_at_k(3) > 0.3
    assert r.citation_correctness(3) > 0.5


# ---------------- llm ----------------
def test_llm_template_fallback():
    from llm import client as C
    C._probe_cache.clear()
    C._probe_cache[C.OLLAMA_BASE] = False
    # force template path regardless of local Ollama server status
    s = C.summarize_text("Patient has chest pain. Plan: ECG advised.")
    assert "chest pain" in s
    rep = {"conditions": [{"condition": "cardiac_risk", "score": 0.8,
                           "risk_level": "HIGH", "confidence": 0.8,
                           "top_factors": ["sbp", "ldl"]}],
           "medication_review": {"flags": []}, "conflicts": [],
           "evidence": [{"citation": "[C01] test"}]}
    d = C.draft_summary(rep)
    assert "cardiac_risk" in d and "HIGH" in d


# ---------------- medication ----------------
def test_medication_flags():
    from agents.medication_data import check_contraindications, check_interactions
    flags = check_interactions(["atorvastatin", "clarithromycin"])
    assert any("clarithromycin" in f["drugs"] for f in flags)
    assert check_interactions(["metformin"]) == []
    c = check_contraindications(["lisinopril"], ["pregnancy"])
    assert len(c) == 1 and c[0]["severity"] == "high"


# ---------------- agents + orchestrator ----------------
def _demo_patient():
    from scripts.showcase import get_showcase_patients
    return get_showcase_patients(get_ctx())[0]


def test_agents_produce_sections():
    from agents.orchestrator import SequentialOrchestrator
    ctx = get_ctx()
    orch = SequentialOrchestrator(ctx)
    p = _demo_patient()
    from agents.base import Trace
    trace = Trace()
    state = {"patient": p}
    for key in ["risk", "image", "research", "medication"]:
        state.update(orch.agents[key].run(state, trace))
    assert 0 <= state["risk"]["combined_score"] <= 1
    assert state["image"]["finding"]
    assert len(state["research"]["evidence"]) > 0
    assert isinstance(state["medication_review"]["flags"], list)


def test_orchestrator_end_to_end():
    from agents.orchestrator import get_orchestrator
    ctx = get_ctx()
    orch = get_orchestrator(ctx)
    p = _demo_patient()
    res = orch.run(p)
    rep = res["state"]["fused_report"]
    assert len(rep["conditions"]) == 3
    assert len(rep["evidence"]) > 0
    assert "verification" in rep
    assert rep["verification"]["passed"] is True
    assert "narrative" in res["state"]["summary"]
    assert len(res["trace"]) > 10


def test_fusion_conflict():
    from fusion.fusion import fuse_evidence
    state = {
        "patient": {"patient_id": "SYN-T", "age": 50, "sex": "M",
                    "total_chol": 180, "glucose": 95, "hba1c": 5.4, "bmi": 23,
                    "medications": [], "labels": {}},
        "risk": {"combined_score": 0.2, "combined_level": "LOW",
                 "tabular": {"model": "xgb", "score": 0.2},
                 "temporal": {"model": "t", "score": 0.2}, "top_factors": []},
        "image": {"score": 0.8, "finding": "opacity", "model": "m",
                  "backend": "heuristic-fallback"},
        "research": {"evidence": []},
        "medication_review": {"medications": [], "flags": [], "n_flags": 0},
    }
    rep = fuse_evidence(state)
    assert any("CONFLICT" in c for c in rep["conflicts"])


def test_verification_flags_missing_evidence():
    from agents.verification import verify_report
    rep = {"conditions": [{"condition": "x", "risk_level": "LOW",
                           "evidence_refs": []}],
           "conflicts": [], "evidence": [],
           "medication_review": {"flags": []}}
    out, issues = verify_report(rep)
    assert out["verification"]["passed"] is False
    assert len(issues) > 0


# ---------------- explain ----------------
def test_explain_fallback():
    import pandas as pd
    from explain.explain import explain_tabular
    from models.tabular import RandomForestModel
    X, y = _small_xy()
    m = RandomForestModel(seed=0).fit(X, y)
    res = explain_tabular(m, X)
    assert len(res["attributions"]) == X.shape[1]
    assert res["method"] in ("shap-tree", "permutation-fallback")


# ---------------- api ----------------
def test_api():
    try:
        from fastapi.testclient import TestClient
        from api.server import app, FASTAPI_AVAILABLE
    except Exception:
        return  # skip if fastapi missing
    assert FASTAPI_AVAILABLE
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["synthetic_only"] is True
    r = c.post("/analyze", json={"patient_id": "SYN-SHOW-01"})
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["verification"]["passed"] is True


def test_fusion_label_invariance():
    """Outcome labels must never enter an inference result."""
    from fusion.fusion import fuse_evidence
    import copy
    base = {"patient_id": "SYN-TEST", "synthetic": True,
            "glucose": 105, "hba1c": 5.5, "bmi": 24,
            "labels": {"cardiac_risk": 0, "diabetes_risk": 0, "respiratory_flag": 0}}
    state = {"patient": base, "risk": {"combined_score": 0.3},
             "image": {"score": 0.7, "backend": "heuristic-fallback"}}
    first = fuse_evidence(state)
    state["patient"] = copy.deepcopy(base)
    state["patient"]["labels"] = {"cardiac_risk": 1, "diabetes_risk": 1,
                                    "respiratory_flag": 1}
    second = fuse_evidence(state)
    assert first["conditions"] == second["conditions"]
    assert first["conflicts"] == second["conflicts"]
    assert first["model_provenance"]["diabetes"] == "threshold-proxy"


def test_fusion_missing_labels():
    from fusion.fusion import fuse_evidence
    report = fuse_evidence({"patient": {"patient_id": "SYN-EMPTY", "synthetic": True},
                            "risk": {}, "image": {}})
    assert len(report["conditions"]) == 3
    assert report["conditions"][1]["confidence"] is None

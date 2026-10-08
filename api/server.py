"""Thin FastAPI backend demonstrating client-server architecture.

GET  /health    -> service status + component availability
POST /analyze   -> run the full agent pipeline on a patient (by showcase id
                   or inline patient dict) and return the verified report.

The Gradio dashboard (app.py) is the primary demo surface; this API shows
the same pipeline served over HTTP.
"""
import time

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except Exception:
    FastAPI = None
    _FASTAPI_AVAILABLE = False

FASTAPI_AVAILABLE = _FASTAPI_AVAILABLE

if _FASTAPI_AVAILABLE:
    import config
    from agents.context import SystemContext
    from agents.orchestrator import get_orchestrator

    app = FastAPI(title="Clinical Intelligence API",
                  description="Agentic multimodal clinical intelligence — "
                              "SYNTHETIC research demo, not a medical device.")
    _CTX = None

    def _ctx():
        global _CTX
        if _CTX is None:
            _CTX = SystemContext()
            _CTX.train_all(verbose=False)
        return _CTX

    class AnalyzeRequest(BaseModel):
        patient_id: str | None = None
        patient: dict | None = None

    @app.get("/health")
    def health():
        from models import vision as V
        from models.temporal import TORCH_AVAILABLE
        from llm.client import available as llm_backend
        return {
            "status": "ok",
            "synthetic_only": True,
            "disclaimer": config.DISCLAIMER,
            "components": {
                "vision_backend": V.available(),
                "torch_temporal": TORCH_AVAILABLE,
                "llm": llm_backend(),
            },
        }

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest):
        t0 = time.time()
        ctx = _ctx()
        patient = None
        if req.patient:
            patient = req.patient
        else:
            from scripts.showcase import get_showcase_patients
            for p in get_showcase_patients(ctx):
                if p["patient_id"] == (req.patient_id or "SYN-SHOW-01"):
                    patient = p
                    break
            if patient is None:
                return JSONResponse(
                    {"error": "patient not found"}, status_code=404)
        # strip non-serializable image before pipeline, reattach after
        orch = get_orchestrator(ctx)
        result = orch.run(patient)
        report = result["state"]["fused_report"]
        report = _json_safe(report)
        return {
            "patient_id": patient["patient_id"],
            "report": report,
            "trace": result["trace"],
            "orchestrator": result["backend"],
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "disclaimer": config.DISCLAIMER,
        }

    def _json_safe(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape}>"
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

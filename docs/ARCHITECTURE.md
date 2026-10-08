# Architecture — Agentic Multimodal Clinical Intelligence System

> Research demo on 100% synthetic data. Not a medical device.

## Big picture

```
SYNTHETIC PATIENT ─┬─ structured labs/demographics ─┐
                   ├─ longitudinal vitals (4–8 visits)
                   ├─ clinical note (free text)
                   └─ synthetic chest-X-ray-like image
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        RiskAgent            ImageAgent           ResearchAgent
   (tabular XGBoost     (DenseNet121 /      (RAG over bundled
    + temporal LSTM/      heuristic +        evidence KB, cited)
    GRU/CNN/Transformer) Grad-CAM)
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                          MedicationAgent ──► interaction table
                                    │
                                    ▼
                          Fusion layer ──► conflict flags, uncertainty
                                    │
                                    ▼
                    Verification layer ──► every claim needs a citation
                                    │
                                    ▼
                          SummaryAgent ──► Ollama / template narrative
                                    │
                                    ▼
                    Gradio dashboard (6 tabs) + FastAPI /analyze
```

## Module map

| Path | Responsibility |
|---|---|
| `data/synth.py` | Deterministic synthetic cohort: demographics, labs, visit vitals, notes, procedural X-ray-like images |
| `models/tabular.py` | XGBoost (→GBM fallback), RandomForest, LogisticRegression — unified fit/predict/save/load |
| `models/temporal.py` | LSTM, GRU, Temporal-CNN, Transformer (torch); trend-feature fallback when torch absent |
| `models/nlp.py` | Rule-based finding extraction + negation, vital parsing, optional HF NER, LLM summarization |
| `models/vision.py` | DenseNet121 fine-tune + Grad-CAM (torch); heuristic lung-opacity fallback otherwise |
| `rag/` | 42-snippet educational KB, TF-IDF+cosine (FAISS if present), rerank, precision@k eval |
| `llm/` | Ollama-first generation; deterministic template fallback — demo never breaks |
| `agents/` | 5 agents + orchestrator (LangGraph if importable, else tested state machine) + verification |
| `explain/` | SHAP (→permutation fallback), transformer attention viewer, attribution plots |
| `fusion/` | Multimodal fusion: per-condition scores, confidence, conflict flags, uncertainty notes |
| `api/server.py` | Thin FastAPI: `GET /health`, `POST /analyze` |
| `app.py` | Gradio dashboard (primary demo surface) |
| `evals/` | Baselines vs full system vs ablations; writes `docs/EVALS.md` |

## Key design decisions

1. **Import-guarded heavy deps.** Every heavy library (torch, shap, faiss,
   langgraph, transformers) is optional. The app runs fully on
   numpy/pandas/sklearn/gradio with honest fallback labels in the UI
   ("heuristic-fallback", "template summary").
2. **Orchestrator fallback.** LangGraph is attempted first, but the explicit
   sequential state machine is the tested path — identical agent order, state
   shape and trace format. A demo must not depend on an untested code path.
3. **Conflicts surface, never average.** If imaging disagrees with tabular
   risk, the report flags it for clinician review.
4. **Verification layer.** The SummaryAgent only narrates the *verified*
   report; unverified claims are listed as issues, not hidden.
5. **Synthetic-first.** No real patient data anywhere. `docs/DATASETS.md`
   documents the swap path to MIMIC-IV / NIH / CheXpert for real research.

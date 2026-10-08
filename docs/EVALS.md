# Evaluation Results (SYNTHETIC DATA)

Generated from `evals/run_evals.py` — cohort n=1200, seed=42,
agentic eval on full test set (n=240 patients). Runtime 23.6s.

> **Honest framing:** every metric below is computed on procedurally generated
> synthetic patients. They validate that the pipeline trains, fuses modalities,
> retrieves evidence and ablates cleanly — they say NOTHING about real clinical
> performance. Decision thresholds are tuned per configuration (max F1) because
> fused scores are not calibrated to 0.5; the threshold used is shown.
> See docs/SAFETY.md.

## Predictive performance (cardiac_risk)

| configuration | AUROC | AUPRC | F1 | precision | recall | thr |
|---|---|---|---|---|---|---|
| baseline:logistic_regression | 0.7491 | 0.2382 | 0.3019 | 0.2424 | 0.4 | 0.15|
| baseline:xgboost | 0.717 | 0.2084 | 0.2353 | 0.1935 | 0.3 | 0.1|
| temporal-only | 0.9632 | 0.7637 | 0.7442 | 0.6957 | 0.8 | 0.1|
| full-agentic-system | 0.9311 | 0.7067 | 0.6122 | 0.5172 | 0.75 | 0.2|
| ablation:no-temporal | 0.717 | 0.2084 | 0.2353 | 0.1935 | 0.3 | 0.1|
| ablation:no-orchestration | 0.717 | 0.2084 | 0.2353 | 0.1935 | 0.3 | 0.1|

## Ablation notes
- **no-temporal**: tabular XGBoost alone on the same subset.
- **no-orchestration**: single model, no agents/fusion — the pipeline's added
  value is evidence grounding + conflict flags, not just AUROC.
- **no-rag**: identical risk scores, but mean citations drop from
  7.23 to 0.0 — RAG's
  contribution is auditability, measured by citation coverage.
- **no-image**: vision feeds the respiratory assessment; cardiac AUROC is
  unaffected by design (modality separation).

## Retrieval evaluation
- precision@3: 0.3333
- citation correctness@3: 0.875
- backend: tfidf-cosine

## System
- mean end-to-end latency per patient: 97.7 ms
- mean evidence citations per report: 7.23

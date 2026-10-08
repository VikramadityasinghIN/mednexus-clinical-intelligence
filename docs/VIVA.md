# Viva Q&A — expected questions and strong answers

**Q: Why synthetic data instead of real patient data?**
A: Privacy and access. Real clinical data needs ethics approval and
credentialed access (MIMIC-IV, CheXpert). Synthetic data lets us validate the
*entire pipeline* — training, fusion, RAG grounding, ablations — with zero
privacy risk. docs/DATASETS.md documents the exact swap path to real datasets.

**Q: Isn't a synthetic evaluation meaningless?**
A: It's meaningful for *systems* validation, not clinical validation. Our
EVALS.md is honest about this: AUROC on synthetic data proves the pipeline
learns the generative signal and that ablations behave sanely — it does not
prove clinical efficacy. That's stated in the report itself.

**Q: Why five agents instead of one big model?**
A: Separation of concerns and auditability. Each agent owns one modality with
its own evidence; the verification layer checks every claim against a source.
A monolith can't produce the per-agent trace the dashboard shows.

**Q: Why LangGraph *or* a custom orchestrator?**
A: We attempt LangGraph for standards-compliance, but the tested path is an
explicit state machine with identical semantics — a demo must not depend on
an untested code path. Both emit the same trace format.

**Q: How do you handle RAG hallucination?**
A: Three ways: retrieval returns verbatim snippets with IDs; the fusion layer
only cites retrieved text; the verification layer fails the report if any
condition lacks citations. The LLM only *phrases* verified content.

**Q: SHAP vs your permutation fallback?**
A: SHAP TreeExplainer when installed — exact for trees. Otherwise permutation
attribution on the patient's row: shuffle each feature, measure prediction
drop. Same ranking intuition, no dependency.

**Q: What does the temporal model add over the latest visit?**
A: Trend features (slope, variability) and sequence models capture
*deterioration trajectories*. Our ablation (no-temporal) quantifies exactly
how much AUROC that contributes — see EVALS.md.

**Q: Grad-CAM — what does the heatmap actually prove?**
A: It shows *where* the vision model looked, not *what* it diagnosed. It's an
audit tool: if the heatmap glows outside the lungs, you distrust the finding.
That's the correct, limited claim.

**Q: Biggest limitation?**
A: Two: no real patient data, and vision weights are ImageNet-pretrained, not
clinically validated. Both are documented in SAFETY.md and the report.

**Q: What would you do with 6 more months?**
A: Credentialed MIMIC-IV + CheXpert training, clinician review of 200 reports
for groundedness scoring, calibration analysis, and a prospective silent-trial
design. The architecture already supports it — only data adapters change.

**Q: Why not just use GPT-4 / a big API model?**
A: Free/local constraint and reproducibility: Ollama runs offline with pinned
weights; API models change under you and cost per token. The template
fallback also guarantees the demo never breaks without a network.

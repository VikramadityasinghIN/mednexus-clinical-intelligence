# MedNexus AI 2.0 — Investor brief (draft, not validated claims)

## Vision
Evidence-grounded clinical intelligence for clinicians: a workflow platform that assembles structured records, vital-sign trends, clinical text and medical imaging into an auditable review experience. Research demonstration only until validated and appropriately authorized.

## Problem and buyer hypothesis
Clinicians face fragmented data and documentation burden. Potential buyers: hospitals, diagnostic networks, specialty clinics, and research institutions. These are hypotheses to validate through interviews; there is no demonstrated revenue or traction in this repository.

## Initial wedge
Clinician-facing case review and evidence synthesis on synthetic or properly licensed de-identified records. Do not market synthetic-risk predictions as clinically accurate.

## Product differentiation to validate
1. Modality-specific provenance and explicit fallback labeling.
2. Source-linked evidence and auditable agent traces.
3. Local-first inference for sensitive research datasets.
4. Model comparisons and leakage-resistant evaluation.
5. Human review and uncertainty surfaced prominently.

## Business model hypotheses
Pilot fee for a bounded research deployment, followed by institutional subscription or per-seat licensing. Prices, savings, and ROI require actual customer discovery and time-motion studies.

## $1M illustrative allocation (USD, not a quotation)
- 30% engineering and infrastructure: $300,000
- 25% clinical validation and research partnerships: $250,000
- 15% privacy, security, quality and regulatory advice: $150,000
- 15% go-to-market and pilot operations: $150,000
- 10% data licensing and computing: $100,000
- 5% contingency: $50,000
Total: $1,000,000. Milestone-based release recommended.

## Milestones before serious fundraising
1. Correct inference leakage and establish reproducible offline tests.
2. Choose one clinically meaningful target and prespecified evaluation protocol.
3. Obtain appropriate dataset rights and conduct patient-level splits, temporal holdout and external validation where feasible.
4. Run clinician usability interviews and recruit a clinical adviser.
5. Pilot with institutional oversight; document time savings and failure cases.
6. Commission regulatory classification and security/privacy assessment in target jurisdictions.
7. Produce a transparent data room: code, model cards, dataset cards, risk register, financial plan, evidence of pilot interest.

## Risks and mitigations
- Clinical harm: human review, no autonomous treatment, visible uncertainty.
- Regulatory: intended-use analysis by qualified counsel; do not assume exempt status.
- Data access: comply with dataset agreements, never distribute credentialed datasets.
- LLM hallucinations: citations, abstention, clinician review, adversarial testing.
- Model generalization: external validation and calibration before any clinical claims.
- Security: authentication, authorization, encryption, audit trails, penetration testing before production.

## Current status (accurate as of this code package)
Existing Gradio/FastAPI research demo, synthetic data generation, tabular/temporal/vision components, agents, RAG and evaluations. Leakage regression fixes added in 2.0. Not clinically validated, not regulatory cleared, not production hardened, no verified customers or revenue.

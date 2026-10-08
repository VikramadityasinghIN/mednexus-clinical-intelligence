# MedNexus AI 2.0 — hardening release

## Changes implemented
- Removed respiratory ground-truth label from the inference-time respiratory score.
- Removed label-dependent branching in multimodal conflict logic.
- Added a regression test verifying that flipping all ground-truth labels does not change inference conditions or conflict flags.
- Added test for label-free patient inputs.
- Added explicit per-condition model provenance for proxy vs trained vs image engines.
- Replaced uncalibrated numerical `confidence` displays with `null` to avoid implying statistical confidence without calibration.
- Added investor brief, validation and commercial milestones.

## Remaining work / nonclaims
- Diabetes is a threshold proxy, not a trained or validated clinical probability.
- Image scores may come from a synthetic-image heuristic fallback.
- No real-world clinical accuracy, funding, market traction, or regulatory approval is claimed.
- Existing RAG verification still checks presence of sources, not factual entailment; source-linked claim verification must be implemented before external deployment.
- No production authentication, clinical governance, or institutional integration has been implemented.
- For MIMIC and other credentialed datasets, do not commit patient data or upload it to noncompliant external LLM services.

## Validation gate
Before any patient-facing or clinician-operational deployment: lock intended use, perform privacy/security review, data access review, external validation, human-factors evaluation, and appropriate regulatory assessment.

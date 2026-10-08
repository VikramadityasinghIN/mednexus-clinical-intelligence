# SAFETY.md — Safety, Privacy & Responsible Use

## Standing position
This system is a **clinical decision-support RESEARCH DEMO** built on
**100% synthetic data**. It is not a medical device. It does not diagnose,
triage, prescribe, or replace professional clinical judgment. Every UI tab and
every generated report carries this disclaimer.

## Data
- All patients, vitals, notes and images are procedurally generated
  (`data/synth.py`, seeded). No real patient data was used, stored, or
  transmitted at any point.
- Synthetic X-ray images are geometric fakes (ellipses, arcs, blobs) — they
  exercise the vision pipeline and nothing more.

## Model limitations (must state in any presentation)
1. **No clinical validation.** Metrics in docs/EVALS.md are on synthetic data;
   they validate the software pipeline, not medical correctness.
2. **Vision weights are not clinical.** DenseNet121 starts from ImageNet
   pretraining; fine-tuning here is on synthetic images. Real deployment needs
   CheXpert/MIMIC-CXR training and radiologist review.
3. **Knowledge base is educational.** The 42 RAG snippets are demo summaries,
   not clinical guidelines. A real build must license actual guidelines.
4. **Medication table is illustrative.** The ~30 interaction entries are
   educational; a real system must query an approved drug database.
5. **LLM outputs are unverified prose.** The LLM only phrases
   verification-passed content; treat its wording as draft assistance.

## Intended use
- Classroom demonstration, viva examination, résumé project.
- Architecture reference for real research with credentialed datasets.

## Prohibited use
- Do not present outputs as medical advice to any person.
- Do not connect to real patient records without institutional ethics approval
  and data-governance review.
- Do not remove the disclaimer banners.

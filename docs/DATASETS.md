# DATASETS.md — swapping synthetic data for real public datasets

The pipeline is data-source agnostic: every module consumes the patient-dict
schema from `data/synth.py`. To go real, write an adapter that yields the
same schema from a public dataset. Suggested sources:

## 1. MIMIC-IV (clinical + temporal) — credentialed
- **What:** ICU/hospital records: labs, vitals time series, notes, meds.
- **Access:** PhysioNet credentialed access — requires CITI training + data
  use agreement (free, 1–2 weeks approval).
- **URL:** https://physionet.org/content/mimiciv/
- **License:** PhysioNet Data Use Agreement (research only, no redistribution).
- **Adapter notes:** map `labevents` → labs dict, `chartevents` → visits,
  `prescriptions` → medications. De-identified already.

## 2. MIMIC-IV demo subset — open
- 100-patient open demo, no credentialing. Ideal for a first real-data run.
- Same URL as above, "mimic-iv-demo" files.

## 3. NIH ChestX-ray14 (imaging) — open
- **What:** 112,120 frontal X-rays with 14 finding labels.
- **Access:** Public download, no login.
  https://nihcc.app.box.com/v/ChestXray-NIHCC
- **License:** Public release (check current terms on the page).
- **Adapter notes:** `Data_Entry_2017.csv` → binary abnormal label or
  multi-label head (see `models/vision.py`). `scripts/fetch_public_xray_sample.py`
  documents the manual steps.

## 4. CheXpert (Stanford, imaging) — gated research
- **What:** 224,316 chest radiographs, uncertainty-labeled.
- **Access:** Research application + agreement.
  https://stanfordmlgroup.github.io/competitions/chexpert/
- **License:** Stanford research use agreement.

## 5. MedQA / PubMedQA (RAG corpus) — open
- Replace the 42-snippet demo KB with real QA pairs or guideline excerpts.
- MedQA: https://github.com/jindongwang/MedQA (research use).

## Governance checklist before using real data
- [ ] Ethics/IRB approval from your institution
- [ ] Dataset license + DUA signed and filed
- [ ] No real data committed to git or zipped into deliverables
- [ ] Re-run `evals/run_evals.py` and regenerate `docs/EVALS.md` on real data
- [ ] Clinician review of at least 100 reports before any claim of utility

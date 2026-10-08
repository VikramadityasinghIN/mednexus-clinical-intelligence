"""Synthetic patient data generator.

Produces a fully synthetic, deterministic (seeded) cohort with four
modalities per patient:
  - structured demographics + labs
  - longitudinal vitals across 4-8 visits (time series)
  - templated clinical notes with realistic phrasing
  - synthetic chest-X-ray-like images (numpy-generated, clearly fake)

Labels: cardiac_risk, diabetes_risk, respiratory_flag — generated from a
hidden probabilistic model so ML models have real signal to learn.

EVERYTHING here is synthetic. Nothing is derived from real patients.
"""
import numpy as np
import pandas as pd

FIRST_NAMES_M = ["Rajesh", "Amit", "Vikram", "Suresh", "Arjun", "Rahul", "Karan", "Deepak"]
FIRST_NAMES_F = ["Priya", "Anita", "Meera", "Kavita", "Sunita", "Neha", "Pooja", "Divya"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Iyer", "Khan", "Gupta", "Reddy", "Nair"]

CONDITION_LABELS = ["cardiac_risk", "diabetes_risk", "respiratory_flag"]

TABULAR_FEATURES = [
    "age", "sex_male",
    "total_chol", "hdl", "ldl", "glucose", "hba1c", "bmi", "sbp", "dbp",
    "smoking", "diabetes", "family_history", "exercise_low",
]

VISIT_FEATURES = ["hr", "sbp", "dbp", "spo2", "weight"]
MAX_VISITS = 8

SYMPTOM_POOL = {
    "cardiac": ["chest discomfort on exertion", "palpitations", "shortness of breath",
                "ankle swelling", "fatigue", "dizziness"],
    "diabetes": ["increased thirst", "frequent urination", "blurred vision",
                 "slow wound healing", "tingling in feet"],
    "respiratory": ["persistent cough", "wheezing", "breathlessness at night",
                    "chest tightness"],
    "general": ["no acute complaints", "routine follow-up", "mild headache"],
}

NOTE_TEMPLATES = [
    "{age}-year-old {sex_word} presents for {reason}. Reports {symptoms}. "
    "Vitals stable. Labs show {labs}. {plan}",
    "{sex_word_cap} aged {age} with history of {history} here for {reason}. "
    "Patient notes {symptoms}. Examination unremarkable. {plan}",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_patients(n=1500, seed=42, with_images=True):
    """Generate n synthetic patients. Returns list of patient dicts.

    with_images=False skips pixel generation (keeps image_label) for
    large training cohorts; use generate_image_cohort() for vision data.
    """
    rng = np.random.RandomState(seed)
    patients = []
    for i in range(n):
        p = _generate_one(rng, i, with_images=with_images)
        patients.append(p)
    return patients


def generate_image_cohort(n=600, seed=123):
    """Stream (image, label) pairs for vision training. Generator."""
    rng = np.random.RandomState(seed)
    for _ in range(n):
        abnormal = bool(rng.rand() < 0.5)
        img, label = generate_synthetic_xray(rng, abnormal=abnormal)
        yield img, label


def _generate_one(rng, i, with_images=True):
    sex_male = int(rng.rand() < 0.52)
    age = int(np.clip(rng.normal(55, 14), 25, 90))
    sex_word = "male" if sex_male else "female"

    smoking = int(rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15]))  # 0 never,1 former,2 current
    diabetes = int(rng.rand() < 0.18 + 0.002 * max(0, age - 50))
    family_history = int(rng.rand() < 0.3)
    exercise_low = int(rng.rand() < 0.45)
    bmi = float(np.clip(rng.normal(27, 5), 16, 48))

    total_chol = float(np.clip(rng.normal(205, 38), 120, 340))
    hdl = float(np.clip(rng.normal(52 - 6 * sex_male, 12), 25, 100))
    ldl = float(np.clip(total_chol - hdl - 40 + rng.normal(0, 10), 60, 250))
    glucose = float(np.clip(rng.normal(100 + 25 * diabetes, 22), 70, 320))
    hba1c = float(np.clip(rng.normal(5.6 + 1.6 * diabetes, 0.7), 4.5, 13))
    sbp = float(np.clip(rng.normal(128 + 0.35 * max(0, age - 50), 16), 95, 210))
    dbp = float(np.clip(rng.normal(82 + 0.15 * max(0, age - 50), 10), 55, 130))

    # Hidden risk model (this is what ML must rediscover)
    logit = (
        -4.2
        + 0.045 * (age - 50)
        + 0.012 * (total_chol - 200)
        - 0.03 * (hdl - 50)
        + 0.02 * (sbp - 130)
        + 0.02 * (glucose - 100)
        + 0.5 * smoking
        + 0.6 * diabetes
        + 0.5 * family_history
        + 0.3 * exercise_low
        + 0.02 * (bmi - 25)
    )
    cardiac_prob = _sigmoid(logit + rng.normal(0, 0.35))
    cardiac_risk = int(rng.rand() < cardiac_prob)

    d_logit = -3.0 + 0.03 * (age - 50) + 0.05 * (glucose - 100) + 0.8 * (hba1c - 5.6) + 0.04 * (bmi - 25)
    diabetes_risk = int(rng.rand() < _sigmoid(d_logit + rng.normal(0, 0.4)))

    r_logit = -3.5 + 0.9 * smoking + (0.8 if age > 65 else 0.0)
    respiratory_flag = int(rng.rand() < _sigmoid(r_logit + rng.normal(0, 0.4)))

    # Longitudinal visits
    n_visits = int(rng.randint(4, MAX_VISITS + 1))
    visits = []
    base_hr = rng.normal(76 + 6 * cardiac_risk, 8)
    base_w = rng.normal(72 + 0.6 * (bmi - 25), 6)
    for v in range(n_visits):
        drift = 0.02 * v * (1 if cardiac_risk else -0.3)  # worsening trend if at risk
        visits.append({
            "visit_no": v + 1,
            "hr": float(np.clip(rng.normal(base_hr * (1 + drift), 4), 45, 130)),
            "sbp": float(np.clip(rng.normal(sbp * (1 + 0.5 * drift), 6), 95, 210)),
            "dbp": float(np.clip(rng.normal(dbp * (1 + 0.5 * drift), 5), 55, 130)),
            "spo2": float(np.clip(rng.normal(97.5 - 1.5 * respiratory_flag, 1.2), 88, 100)),
            "weight": float(np.clip(rng.normal(base_w + 0.3 * v * (1 if bmi > 30 else 0), 1.5), 40, 160)),
        })

    # Medications (plausible given profile)
    medications = []
    if total_chol > 220 or cardiac_risk:
        medications.append("atorvastatin")
    if sbp > 140:
        medications.append(rng.choice(["amlodipine", "metoprolol", "lisinopril"]))
    if diabetes or glucose > 126:
        medications.append("metformin")
    if respiratory_flag:
        medications.append("salbutamol inhaler")
    if rng.rand() < 0.2:
        medications.append("aspirin")

    note = _generate_note(rng, age, sex_word, cardiac_risk, diabetes_risk,
                          respiratory_flag, smoking, diabetes, total_chol, sbp, glucose)

    abnormal_flag = bool(respiratory_flag or (cardiac_risk and rng.rand() < 0.5))
    image_label = int(abnormal_flag)
    if with_images:
        image, _ = generate_synthetic_xray(rng, abnormal=abnormal_flag)
    else:
        image = None  # pixels regenerated on demand; label kept

    name = f"{rng.choice(FIRST_NAMES_M if sex_male else FIRST_NAMES_F)} {rng.choice(LAST_NAMES)}"

    return {
        "patient_id": f"SYN-{i:05d}",
        "synthetic": True,
        "name": name,
        "age": age,
        "sex": "M" if sex_male else "F",
        "sex_male": sex_male,
        "total_chol": total_chol, "hdl": hdl, "ldl": ldl,
        "glucose": glucose, "hba1c": hba1c, "bmi": bmi,
        "sbp": sbp, "dbp": dbp,
        "smoking": smoking, "diabetes": diabetes,
        "family_history": family_history, "exercise_low": exercise_low,
        "visits": visits,
        "medications": medications,
        "note": note,
        "image": image,
        "image_label": int(image_label),
        "labels": {
            "cardiac_risk": cardiac_risk,
            "diabetes_risk": diabetes_risk,
            "respiratory_flag": respiratory_flag,
        },
        "cardiac_prob_true": float(cardiac_prob),
    }


def _generate_note(rng, age, sex_word, cardiac, diabetes_r, resp, smoking, diabetes, chol, sbp, glucose):
    if cardiac:
        symptoms = ", ".join(rng.choice(SYMPTOM_POOL["cardiac"], 2, replace=False))
        history, reason = "hypertension and hyperlipidemia", "cardiac evaluation"
    elif diabetes_r or diabetes:
        symptoms = ", ".join(rng.choice(SYMPTOM_POOL["diabetes"], 2, replace=False))
        history, reason = "type 2 diabetes", "diabetes review"
    elif resp:
        symptoms = ", ".join(rng.choice(SYMPTOM_POOL["respiratory"], 2, replace=False))
        history, reason = "recurrent bronchitis", "respiratory assessment"
    else:
        symptoms = ", ".join(rng.choice(SYMPTOM_POOL["general"], 2, replace=False))
        history, reason = "no significant past history", "annual health check"

    if rng.rand() < 0.25:  # negation example for the NLP module
        symptoms += ". Denies chest pain at rest"
    labs = f"total cholesterol {chol:.0f} mg/dL, BP {sbp:.0f} mmHg systolic, fasting glucose {glucose:.0f} mg/dL"
    plan = ("Plan: lifestyle counselling, repeat labs in 3 months."
            if not cardiac else
            "Plan: ECG and lipid panel advised; cardiology referral suggested.")
    tpl = rng.choice(NOTE_TEMPLATES)
    sex_cap = sex_word.capitalize()
    return tpl.format(age=age, sex_word=sex_word, sex_word_cap=sex_cap,
                      reason=reason, symptoms=symptoms, labs=labs,
                      history=history, plan=plan)


def generate_synthetic_xray(rng, abnormal=False, size=224):
    """Procedurally generate a fake chest-X-ray-like image.

    Clearly NOT a real radiograph: smooth geometric lung fields, rib arcs,
    optional synthetic opacity blob / nodule. Exists only to exercise the
    vision pipeline end-to-end.
    Returns (image float32 0..1, label int).
    """
    img = np.full((size, size), 0.10, dtype=np.float32)  # dark background
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size // 2

    # Body / soft tissue region
    body = ((xx - cx) ** 2 / (size * 0.42) ** 2 + (yy - size * 0.52) ** 2 / (size * 0.46) ** 2) < 1
    img[body] = 0.35

    # Two lung fields (darker ellipses)
    for side in (-1, 1):
        lx = cx + side * int(size * 0.20)
        lung = ((xx - lx) ** 2 / (size * 0.13) ** 2 + (yy - size * 0.50) ** 2 / (size * 0.30) ** 2) < 1
        img[lung] = 0.06 + rng.rand() * 0.02

    # Spine (vertical bar)
    spine = (np.abs(xx - cx) < size * 0.035) & (yy > size * 0.15) & (yy < size * 0.9)
    img[spine] = 0.55

    # Rib arcs (light arcs across lungs)
    for k in range(5):
        y0 = int(size * (0.28 + 0.11 * k))
        rib = (np.abs(yy - y0 - 0.0008 * (xx - cx) ** 2) < 2) & body
        img[rib] = np.maximum(img[rib], 0.42)

    # Clavicles
    for side in (-1, 1):
        clav = (np.abs(yy - (size * 0.22 + 0.0006 * (xx - cx - side * size * 0.2) ** 2)) < 2.5) & \
               (np.abs(xx - (cx + side * size * 0.22)) < size * 0.16)
        img[clav] = np.maximum(img[clav], 0.5)

    label = 0
    if abnormal:
        label = 1
        # Synthetic opacity: soft white blob in a random lung
        side = rng.choice([-1, 1])
        bx = cx + side * int(size * rng.uniform(0.12, 0.28))
        by = int(size * rng.uniform(0.35, 0.65))
        br = int(size * rng.uniform(0.05, 0.10))
        blob = ((xx - bx) ** 2 + (yy - by) ** 2) < br ** 2
        img[blob] = np.minimum(img[blob] + 0.45, 0.85)
        if rng.rand() < 0.5:  # occasional nodule
            nx, ny = int(rng.uniform(0.3, 0.7) * size), int(rng.uniform(0.3, 0.7) * size)
            nod = ((xx - nx) ** 2 + (yy - ny) ** 2) < (size * 0.02) ** 2
            img[nod] = 0.9

    # Film grain
    img += rng.normal(0, 0.015, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32), label


def patients_to_frame(patients):
    """Flatten patients to a tabular DataFrame + label dict for sklearn."""
    rows = []
    for p in patients:
        row = {"patient_id": p["patient_id"]}
        for f in TABULAR_FEATURES:
            row[f] = p[f]
        for lab in CONDITION_LABELS:
            row[lab] = p["labels"][lab]
        rows.append(row)
    return pd.DataFrame(rows)


def patients_to_sequences(patients, max_visits=MAX_VISITS):
    """Return (n, max_visits, n_features) array + mask for temporal models."""
    n = len(patients)
    X = np.zeros((n, max_visits, len(VISIT_FEATURES)), dtype=np.float32)
    mask = np.zeros((n, max_visits), dtype=np.float32)
    for i, p in enumerate(patients):
        for v, visit in enumerate(p["visits"][:max_visits]):
            X[i, v] = [visit[f] for f in VISIT_FEATURES]
            mask[i, v] = 1.0
    return X, mask

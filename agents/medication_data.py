"""Bundled medication interaction / contraindication table.

~30 realistic entries used by the MedicationAgent. Educational demo content —
a real deployment would query an approved drug database (see docs/DATASETS.md).
Each entry: drugs involved, severity, mechanism/effect, recommendation.
"""
INTERACTIONS = [
    {"drugs": ["atorvastatin", "clarithromycin"], "severity": "high",
     "effect": "Increased statin levels -> higher risk of muscle injury (rhabdomyolysis).",
     "advice": "Consider pausing statin or using an alternative antibiotic."},
    {"drugs": ["atorvastatin", "grapefruit juice"], "severity": "moderate",
     "effect": "Grapefruit inhibits CYP3A4, raising statin blood levels.",
     "advice": "Limit grapefruit to small amounts or avoid."},
    {"drugs": ["metoprolol", "salbutamol inhaler"], "severity": "moderate",
     "effect": "Non-selective beta-blockade can blunt bronchodilator effect and provoke bronchospasm.",
     "advice": "Prefer cardioselective beta-blocker; monitor breathing."},
    {"drugs": ["lisinopril", "spironolactone"], "severity": "high",
     "effect": "Additive potassium retention -> risk of dangerous hyperkalemia.",
     "advice": "Monitor potassium closely; avoid combination if possible."},
    {"drugs": ["lisinopril", "ibuprofen"], "severity": "moderate",
     "effect": "NSAIDs reduce ACE-inhibitor effectiveness and strain kidneys.",
     "advice": "Use paracetamol instead where possible."},
    {"drugs": ["aspirin", "ibuprofen"], "severity": "moderate",
     "effect": "Ibuprofen can block aspirin's antiplatelet effect if taken together.",
     "advice": "Take aspirin at least 30 min before ibuprofen."},
    {"drugs": ["aspirin", "warfarin"], "severity": "high",
     "effect": "Greatly increased bleeding risk.",
     "advice": "Only combined under specialist supervision."},
    {"drugs": ["metformin", "contrast dye"], "severity": "moderate",
     "effect": "Iodinated contrast + metformin raises lactic acidosis risk in kidney impairment.",
     "advice": "Pause metformin around contrast procedures per protocol."},
    {"drugs": ["metformin", "alcohol"], "severity": "moderate",
     "effect": "Heavy alcohol increases lactic acidosis risk with metformin.",
     "advice": "Avoid binge drinking."},
    {"drugs": ["amlodipine", "simvastatin"], "severity": "moderate",
     "effect": "Amlodipine raises simvastatin levels; limit simvastatin dose.",
     "advice": "Max simvastatin 20 mg with amlodipine, or switch statin."},
    {"drugs": ["amlodipine", "grapefruit juice"], "severity": "low",
     "effect": "Mild increase in amlodipine levels.",
     "advice": "No major restriction; monitor for ankle swelling."},
    {"drugs": ["furosemide", "lisinopril"], "severity": "moderate",
     "effect": "First-dose hypotension risk when starting ACE inhibitor on diuretic.",
     "advice": "Start low, monitor blood pressure."},
    {"drugs": ["furosemide", "digoxin"], "severity": "high",
     "effect": "Diuretic-induced low potassium increases digoxin toxicity risk.",
     "advice": "Monitor potassium and digoxin levels."},
    {"drugs": ["spironolactone", "potassium supplements"], "severity": "high",
     "effect": "Severe hyperkalemia risk.",
     "advice": "Avoid combination; monitor potassium."},
    {"drugs": ["warfarin", "ibuprofen"], "severity": "high",
     "effect": "Markedly increased bleeding risk.",
     "advice": "Avoid NSAIDs; use paracetamol."},
    {"drugs": ["clopidogrel", "omeprazole"], "severity": "moderate",
     "effect": "Omeprazole may reduce clopidogrel activation via CYP2C19.",
     "advice": "Consider pantoprazole instead."},
    {"drugs": ["atorvastatin", "warfarin"], "severity": "moderate",
     "effect": "Statins can modestly raise INR in warfarin users.",
     "advice": "Check INR after starting statin."},
    {"drugs": ["metoprolol", "verapamil"], "severity": "high",
     "effect": "Combined heart-rate and conduction slowing -> bradycardia/heart block.",
     "advice": "Avoid combination or use with cardiac monitoring."},
    {"drugs": ["lisinopril", "pregnancy"], "severity": "high",
     "effect": "ACE inhibitors are teratogenic, especially in 2nd/3rd trimester.",
     "advice": "Contraindicated in pregnancy; switch drug class."},
    {"drugs": ["statin", "pregnancy"], "severity": "high",
     "effect": "Statins contraindicated in pregnancy.",
     "advice": "Stop statin if pregnancy planned or confirmed."},
    {"drugs": ["salbutamol inhaler", "metoprolol"], "severity": "moderate",
     "effect": "Beta-blocker may reduce rescue inhaler effectiveness.",
     "advice": "Monitor asthma control."},
    {"drugs": ["insulin", "alcohol"], "severity": "moderate",
     "effect": "Alcohol increases hypoglycemia risk with insulin.",
     "advice": "Eat when drinking; monitor glucose."},
    {"drugs": ["metformin", "cimetidine"], "severity": "low",
     "effect": "Cimetidine can raise metformin levels modestly.",
     "advice": "Monitor for GI upset."},
    {"drugs": ["amlodipine", "alcohol"], "severity": "low",
     "effect": "Additive blood-pressure lowering; dizziness possible.",
     "advice": "Rise slowly; limit alcohol."},
    {"drugs": ["aspirin", "alcohol"], "severity": "moderate",
     "effect": "Increased stomach irritation and bleeding risk.",
     "advice": "Take with food; limit alcohol."},
    {"drugs": ["furosemide", "alcohol"], "severity": "moderate",
     "effect": "Dehydration and low blood pressure risk.",
     "advice": "Maintain fluids; monitor."},
    {"drugs": ["digoxin", "antacids"], "severity": "low",
     "effect": "Antacids can reduce digoxin absorption.",
     "advice": "Separate doses by 2 hours."},
    {"drugs": ["levothyroxine", "calcium"], "severity": "moderate",
     "effect": "Calcium binds levothyroxine, reducing absorption.",
     "advice": "Take levothyroxine 4 hours apart from calcium."},
    {"drugs": ["iron supplements", "levothyroxine"], "severity": "moderate",
     "effect": "Iron reduces levothyroxine absorption.",
     "advice": "Separate by at least 4 hours."},
    {"drugs": ["potassium supplements", "amiloride"], "severity": "high",
     "effect": "Dangerous hyperkalemia.",
     "advice": "Avoid combination."},
]

# Condition-based contraindication flags
CONTRAINDICATIONS = [
    {"drug": "ibuprofen", "condition": "kidney disease",
     "note": "NSAIDs can worsen kidney function."},
    {"drug": "metoprolol", "condition": "asthma",
     "note": "May provoke bronchospasm; use cardioselective agent with caution."},
    {"drug": "lisinopril", "condition": "pregnancy",
     "note": "Teratogenic — contraindicated."},
    {"drug": "statin", "condition": "pregnancy",
     "note": "Contraindicated in pregnancy."},
    {"drug": "aspirin", "condition": "peptic ulcer",
     "note": "Raises bleeding risk; needs gastroprotection review."},
]


def check_interactions(medications):
    """Return list of interaction flags for a medication list."""
    meds = {m.lower().strip() for m in (medications or [])}
    flags = []
    for entry in INTERACTIONS:
        drugs = {d.lower() for d in entry["drugs"]}
        # match if >=2 listed drugs present, or one listed drug + any med
        # containing its name as substring (e.g. "statin" matches "atorvastatin")
        matched = set()
        for d in drugs:
            for m in meds:
                if d == m or d in m or m in d:
                    matched.add(d)
                    break
        if len(matched) >= 2 or (len(drugs) == 2 and len(matched) == 2):
            flags.append({
                "type": "interaction",
                "drugs": entry["drugs"],
                "severity": entry["severity"],
                "effect": entry["effect"],
                "advice": entry["advice"],
            })
    # de-dup by drug pair
    seen, uniq = set(), []
    for f in flags:
        key = tuple(sorted(f["drugs"]))
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def check_contraindications(medications, conditions):
    meds = {m.lower().strip() for m in (medications or [])}
    conds = {c.lower().strip() for c in (conditions or [])}
    flags = []
    for entry in CONTRAINDICATIONS:
        drug_hit = any(entry["drug"] in m or m in entry["drug"] for m in meds)
        cond_hit = any(entry["condition"] in c for c in conds)
        if drug_hit and cond_hit:
            flags.append({"type": "contraindication",
                          "drug": entry["drug"],
                          "condition": entry["condition"],
                          "note": entry["note"],
                          "severity": "high"})
    return flags

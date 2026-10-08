"""Clinical NLP: symptom/finding extraction, negation handling, summarization.

Rule-based extractor works everywhere (zero heavy deps). An optional
HuggingFace transformers NER pipeline is used automatically when available.
Summarization goes through the llm/ layer (Ollama-first, template fallback).
"""
import re

try:
    from transformers import pipeline as _hf_pipeline
    _HF_AVAILABLE = True
except Exception:
    _hf_pipeline = None
    _HF_AVAILABLE = False

HF_AVAILABLE = _HF_AVAILABLE

# Canonical finding -> synonyms / trigger phrases
FINDINGS = {
    "chest_pain": [r"chest (pain|discomfort|tightness|pressure)", r"angina"],
    "dyspnea": [r"short(ness)? of breath", r"dyspn(ea|oea)", r"breathlessness"],
    "palpitations": [r"palpitations?", r"racing heart"],
    "edema": [r"(ankle|leg|pedal) (swelling|edema)", r"\bedema\b"],
    "fatigue": [r"\bfatigue\b", r"\btiredness\b"],
    "dizziness": [r"dizziness", r"light-?headedness"],
    "cough": [r"\bcough\b", r"coughing"],
    "wheeze": [r"wheez(e|ing)"],
    "polyuria": [r"frequent urination", r"polyuria"],
    "polydipsia": [r"increased thirst", r"polydipsia"],
    "blurred_vision": [r"blurred vision"],
    "paresthesia": [r"tingling in (feet|hands)", r"numbness"],
    "hypertension": [r"hypertension", r"high blood pressure", r"\bHTN\b"],
    "hyperlipidemia": [r"hyperlipidemia", r"high cholesterol"],
    "diabetes": [r"\bdiabetes\b", r"\bT2DM\b"],
}

NEGATION_RE = re.compile(
    r"\b(denies|denied|no|without|negative for|ruled out|absence of)\b([\w\s,]{0,40})",
    re.IGNORECASE)

VITAL_RE = {
    "bp": re.compile(r"\bBP\s*(\d{2,3})\s*/\s*(\d{2,3})", re.IGNORECASE),
    "hr": re.compile(r"\bHR\s*(\d{2,3})", re.IGNORECASE),
    "spo2": re.compile(r"SpO2\s*(\d{2,3})", re.IGNORECASE),
}

_hf_ner = None


def _get_hf_ner():
    global _hf_ner
    if not _HF_AVAILABLE or _hf_ner is not None:
        return _hf_ner
    try:
        _hf_ner = _hf_pipeline("ner", model="d4data/biomedical-ner-all",
                               aggregation_strategy="simple")
    except Exception:
        _hf_ner = False
    return _hf_ner if _hf_ner else None


def extract_findings(note):
    """Return list of {finding, negated, span} for a clinical note."""
    text = note or ""
    lower = text.lower()
    out = []
    for finding, patterns in FINDINGS.items():
        for pat in patterns:
            for m in re.finditer(pat, lower):
                span = m.group(0)
                negated = _is_negated(lower, m.start())
                out.append({"finding": finding, "negated": negated,
                            "span": text[m.start():m.end()]})
    # de-dup identical findings keeping negation info
    seen = {}
    for e in out:
        key = (e["finding"], e["negated"])
        if key not in seen:
            seen[key] = e
    out = list(seen.values())

    # Optional transformer NER as a second opinion (import-guarded)
    ner = _get_hf_ner()
    hf_entities = []
    if ner:
        try:
            for ent in ner(text[:2000]):
                hf_entities.append({"entity": ent.get("entity_group"),
                                    "word": ent.get("word"),
                                    "score": float(ent.get("score", 0))})
        except Exception:
            pass
    return {"rule_based": out, "hf_ner": hf_entities,
            "hf_used": bool(hf_entities)}


def _is_negated(lower_text, pos, window=45):
    """Check a window before the match for negation cues."""
    start = max(0, pos - window)
    chunk = lower_text[start:pos]
    return bool(re.search(r"\b(denies|denied|no|without|negative for|ruled out)\b", chunk))


def extract_vitals(note):
    """Pull vital-sign mentions out of free text."""
    out = {}
    m = VITAL_RE["bp"].search(note or "")
    if m:
        out["bp"] = (int(m.group(1)), int(m.group(2)))
    m = VITAL_RE["hr"].search(note or "")
    if m:
        out["hr"] = int(m.group(1))
    m = VITAL_RE["spo2"].search(note or "")
    if m:
        out["spo2"] = int(m.group(1))
    return out


def summarize_note(note, max_sentences=3):
    """Summarize a note via the LLM layer (Ollama-first, template fallback)."""
    from llm.client import summarize_text
    return summarize_text(note, max_sentences=max_sentences)


def highlight_terms(note):
    """Return note text with finding spans marked for UI highlighting."""
    res = extract_findings(note)
    spans = [(e["span"], e["finding"], e["negated"]) for e in res["rule_based"]]
    return spans

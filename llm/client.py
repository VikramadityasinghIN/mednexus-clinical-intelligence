"""LLM layer: Ollama-first (local, free), deterministic template fallback.

`generate(prompt)` uses local Ollama when reachable (probed once, JARVIS-style
via stdlib urllib — no new dependencies). When Ollama is absent the demo NEVER
breaks: `summarize_text` / `draft_summary` fall back to deterministic
template text, and `available()` reports the real backend.
"""
import json
import os
import re
import urllib.request

OLLAMA_BASE = os.environ.get("JARVIS_OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("JARVIS_OLLAMA_MODEL", "qwen3:4b")

_probe_cache = {}


def ollama_available(base=OLLAMA_BASE):
    if base in _probe_cache:
        return _probe_cache[base]
    try:
        req = urllib.request.Request(base + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    _probe_cache[base] = ok
    return ok


def available():
    return "ollama:" + OLLAMA_MODEL if ollama_available() else "template-fallback"


def generate(prompt, system=None, model=OLLAMA_MODEL, timeout=60):
    """Generate text with Ollama. Raises RuntimeError if unreachable."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_BASE + "/api/generate", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode())
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_BASE}: {e}")
    return body.get("response", "").strip()


SYSTEM_CLINICAL = (
    "You are a clinical documentation assistant for a RESEARCH DEMO running on "
    "synthetic data. Write concise, neutral summaries. Never invent facts not "
    "present in the input. End with: 'Synthetic demo data — not medical advice.'"
)


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]


def summarize_text(text, max_sentences=3):
    """Extractive-ish summary: LLM when available, template fallback otherwise."""
    if ollama_available():
        try:
            out = generate(
                f"Summarize the following clinical note in at most {max_sentences} "
                f"sentences, keeping key symptoms, history and plan:\n\n{text}",
                system=SYSTEM_CLINICAL)
            if out:
                return out
        except Exception:
            pass
    # Deterministic fallback: first sentences + key-finding sentence
    sents = _split_sentences(text)
    if not sents:
        return "No note text provided. Synthetic demo data."
    picked = sents[:max_sentences]
    return " ".join(picked) + " [template summary — synthetic demo]"


def draft_summary(report):
    """Draft the SummaryAgent's narrative from the fused report dict."""
    if ollama_available():
        try:
            prompt = ("Write a concise structured clinical summary for this "
                      "synthetic research-demo patient report. Use headings: "
                      "Risk Assessment, Key Findings, Evidence, Medication "
                      "Review, Recommended Next Steps. Keep it factual and "
                      "brief.\n\nReport JSON:\n"
                      + json.dumps(report, indent=1, default=str)[:6000])
            out = generate(prompt, system=SYSTEM_CLINICAL)
            if out:
                return out + "\n\n[Generated with local Ollama — synthetic demo data]"
        except Exception:
            pass
    # Template fallback — always works
    lines = ["CLINICAL INTELLIGENCE SUMMARY (template draft — synthetic demo)"]
    for cond in report.get("conditions", []):
        lines.append(
            f"- {cond['condition']}: {cond['risk_level']} "
            f"(score {cond['score']:.2f}, confidence: not calibrated)")
        for f_ in cond.get("top_factors", [])[:4]:
            lines.append(f"    * {f_}")
    meds = report.get("medication_review", {})
    if meds.get("flags"):
        lines.append("Medication flags:")
        for fl in meds["flags"]:
            lines.append(f"    ! {fl}")
    if report.get("conflicts"):
        lines.append("Conflicts requiring clinician review:")
        for c in report["conflicts"]:
            lines.append(f"    ! {c}")
    lines.append("Evidence: " + "; ".join(
        e.get("citation", "") for e in report.get("evidence", [])[:6]))
    lines.append("Synthetic demo data — not medical advice.")
    return "\n".join(lines)

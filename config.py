"""Global configuration for the Clinical Intelligence system.

All paths resolve relative to the project root. Everything runs on synthetic
data unless the user explicitly swaps in a real dataset (see docs/DATASETS.md).
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Reproducibility
GLOBAL_SEED = 42

# Synthetic cohort defaults
DEFAULT_N_PATIENTS = 1500
DEFAULT_SEED = 42

# Image settings (synthetic chest-X-ray-like images)
IMAGE_SIZE = 224

# Ollama (local, free) — JARVIS-style probing
OLLAMA_BASE = os.environ.get("JARVIS_OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("JARVIS_OLLAMA_MODEL", "qwen3:4b")

# API
API_HOST = "127.0.0.1"
API_PORT = 8123

DISCLAIMER = (
    "RESEARCH DEMO ONLY — all data shown is SYNTHETIC and generated for "
    "software demonstration. This system does not diagnose, treat, or replace "
    "professional clinical judgment."
)

for _d in (ARTIFACTS_DIR,):
    os.makedirs(_d, exist_ok=True)

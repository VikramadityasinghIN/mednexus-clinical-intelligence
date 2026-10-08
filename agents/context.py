"""SystemContext: trains (or loads cached) all models and holds shared
resources for the agents: tabular models, temporal model, vision model,
RAG retriever, and dataset splits. Everything cached under artifacts/.
"""
import os
import pickle

import numpy as np
import pandas as pd

import config
from data.synth import (CONDITION_LABELS, MAX_VISITS, TABULAR_FEATURES,
                        VISIT_FEATURES, generate_patients, patients_to_frame,
                        patients_to_sequences)


class SystemContext:
    def __init__(self, n_patients=1500, seed=42, artifacts_dir=None):
        self.n_patients = n_patients
        self.seed = seed
        self.artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._paths = {
            "tabular": os.path.join(self.artifacts_dir, f"tabular_models_{n_patients}_{seed}.pkl"),
            "temporal": os.path.join(self.artifacts_dir, f"temporal_model_{n_patients}_{seed}.pkl"),
            "vision": os.path.join(self.artifacts_dir, "vision_model.pkl"),
            "splits": os.path.join(self.artifacts_dir, f"splits_{n_patients}_{seed}.pkl"),
        }
        self.tabular_models = {}
        self.temporal_model = None
        # NOTE: artifacts are keyed by (n_patients, seed) so a stale cache
        # from a different environment can never be silently reused.
        self.vision_model = None
        self.retriever = None
        self.splits = None

    def _load_artifact(self, key):
        """Load a cached artifact; return None if missing or unloadable.

        A stale artifact (e.g. pickled with xgboost/torch that is not
        installed in this environment) must never crash startup — the
        caller falls through and retrains with available libraries.
        """
        path = self._paths[key]
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass
            return None

    # ---------------- data ----------------
    def load_data(self):
        from sklearn.model_selection import train_test_split
        if self.splits is not None:
            return self.splits
        cached = self._load_artifact("splits")
        if cached is not None:
            self.splits = cached
            return self.splits
        patients = generate_patients(self.n_patients, self.seed, with_images=False)
        frame = patients_to_frame(patients)
        X = frame[TABULAR_FEATURES]
        idx = np.arange(len(frame))
        tr, te = train_test_split(idx, test_size=0.2, random_state=self.seed)
        X_seq, mask = patients_to_sequences(patients)
        self.splits = {
            "patients": patients,
            "X_train": X.iloc[tr], "X_test": X.iloc[te],
            "mask_train": mask[tr], "mask_test": mask[te],
            "X_seq_train": X_seq[tr], "X_seq_test": X_seq[te],
        }
        for lab in CONDITION_LABELS:
            y = frame[lab].values
            self.splits[f"y_train_{lab}"] = y[tr]
            self.splits[f"y_test_{lab}"] = y[te]
        with open(self._paths["splits"], "wb") as f:
            pickle.dump(self.splits, f)
        return self.splits

    # ---------------- tabular ----------------
    def load_tabular(self, label="cardiac_risk"):
        from models.tabular import MODEL_CLASSES
        if label in self.tabular_models:
            return self.tabular_models[label]
        cached = self._load_artifact("tabular")
        if isinstance(cached, dict):
            self.tabular_models = cached
            if label in self.tabular_models:
                return self.tabular_models[label]
        s = self.load_data()
        trained = {}
        for name, cls in MODEL_CLASSES.items():
            m = cls(seed=self.seed).fit(s["X_train"], s[f"y_train_{label}"])
            trained[name] = m
        self.tabular_models[label] = trained
        with open(self._paths["tabular"], "wb") as f:
            pickle.dump(self.tabular_models, f)
        return trained

    def best_tabular(self, label="cardiac_risk"):
        """Pick the tabular model with best AUROC on the test split."""
        from sklearn.metrics import roc_auc_score
        s = self.load_data()
        trained = self.load_tabular(label)
        best, best_auc, best_name = None, -1, None
        for name, m in trained.items():
            auc = roc_auc_score(s[f"y_test_{label}"],
                                m.predict_proba(s["X_test"]))
            if auc > best_auc:
                best, best_auc, best_name = m, auc, name
        return best_name, best, best_auc

    # ---------------- temporal ----------------
    def load_temporal(self, label="cardiac_risk"):
        if self.temporal_model is not None:
            return self.temporal_model
        cached = self._load_artifact("temporal")
        if cached is not None:
            self.temporal_model = cached
            return self.temporal_model
        from models.temporal import (TORCH_AVAILABLE, TrendFeatureModel,
                                     get_temporal_model)
        s = self.load_data()
        n_features = s["X_seq_train"].shape[2]
        name = "lstm" if TORCH_AVAILABLE else "trend_fallback"
        model = get_temporal_model(name, n_features, seed=self.seed, epochs=12)
        model.fit(s["X_seq_train"], s["mask_train"], s[f"y_train_{label}"])
        self.temporal_model = model
        try:
            with open(self._paths["temporal"], "wb") as f:
                pickle.dump(model, f)
        except Exception:
            pass  # torch models may not pickle cleanly; retrain is fine
        return model

    # ---------------- vision ----------------
    def load_vision(self):
        if self.vision_model is not None:
            return self.vision_model
        cached = self._load_artifact("vision")
        if cached is not None:
            self.vision_model = cached
            return self.vision_model
        from models import vision as V
        self.vision_model = V.get_vision_model()
        return self.vision_model

    # ---------------- rag ----------------
    def load_retriever(self):
        if self.retriever is None:
            from rag.retriever import EvidenceRetriever
            self.retriever = EvidenceRetriever()
        return self.retriever

    # ---------------- helpers ----------------
    def patient_row(self, patient):
        """Tabular feature vector (DataFrame, one row) for a patient dict."""
        row = {f: patient[f] for f in TABULAR_FEATURES}
        return pd.DataFrame([row])

    def patient_sequence(self, patient):
        X, mask = patients_to_sequences([patient], max_visits=MAX_VISITS)
        return X, mask

    def train_all(self, verbose=True):
        log = print if verbose else (lambda *a, **k: None)
        log("Training tabular models...")
        self.load_tabular("cardiac_risk")
        log("Training temporal model...")
        self.load_temporal("cardiac_risk")
        log("Loading vision + RAG...")
        self.load_vision()
        self.load_retriever()
        log("All components ready.")

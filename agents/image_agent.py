"""ImageAgent: chest-X-ray analysis + Grad-CAM heatmap."""
import numpy as np

from .base import Agent


class ImageAgent(Agent):
    name = "ImageAgent"

    def run(self, state, trace):
        patient = state["patient"]
        ctx = self.ctx

        def _do():
            vmodel = ctx.load_vision()
            img = patient.get("image")
            if img is None:
                # regenerate deterministically for showcase patients
                from data.synth import generate_synthetic_xray
                rng = np.random.RandomState(9000 + hash(patient["patient_id"]) % 10**6)
                img, _ = generate_synthetic_xray(
                    rng, abnormal=bool(patient.get("image_label", 0)))
            img = np.asarray(img, dtype=np.float32)
            score = float(vmodel.predict_proba(img[None])[0])
            try:
                hm = vmodel.heatmap(img)
                heatmap = hm[0] if isinstance(hm, tuple) else hm
                heatmap = np.asarray(heatmap, dtype=np.float32)
            except Exception:
                heatmap = None
            finding = ("opacity/nodule-like region detected"
                       if score >= 0.5 else "no focal abnormality detected")
            return {
                "model": vmodel.name,
                "backend": ("torch-densenet121"
                            if not getattr(vmodel, "is_fallback", False)
                            else "heuristic-fallback"),
                "score": round(score, 4),
                "finding": finding,
                "heatmap": heatmap,   # numpy array or None
                "synthetic_image": True,
            }

        result = self._timed(state, trace, "image_analysis", _do,
                             f"xray analysis for {patient['patient_id']}")
        trace.log(self.name, "evidence",
                  f"model={result['model']} finding='{result['finding']}'")
        return {"image": result}

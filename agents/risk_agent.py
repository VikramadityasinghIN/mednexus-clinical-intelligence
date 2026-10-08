"""RiskAgent: tabular + temporal risk scoring with explanations."""
import numpy as np

from .base import Agent


def _level(score):
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MODERATE"
    return "LOW"


class RiskAgent(Agent):
    name = "RiskAgent"

    def run(self, state, trace):
        patient = state["patient"]
        ctx = self.ctx

        def _do():
            s = ctx.load_data()
            model_name, model, auc = ctx.best_tabular("cardiac_risk")
            X1 = ctx.patient_row(patient)
            score = float(model.predict_proba(X1)[0])

            # temporal model score
            tmodel = ctx.load_temporal("cardiac_risk")
            Xs, mask = ctx.patient_sequence(patient)
            tscore = float(tmodel.predict_proba(Xs, mask)[0])

            combined = float(0.65 * score + 0.35 * tscore)
            importances = model.feature_importance()
            top = [{"feature": k, "importance": round(v, 4)}
                   for k, v in list(importances.items())[:6]]
            return {
                "tabular": {"model": model_name, "score": round(score, 4),
                            "level": _level(score), "test_auroc": round(auc, 4)},
                "temporal": {"model": tmodel.name, "score": round(tscore, 4),
                             "level": _level(tscore),
                             "fallback": bool(getattr(tmodel, "is_fallback", False))},
                "combined_score": round(combined, 4),
                "combined_level": _level(combined),
                "top_factors": top,
            }

        result = self._timed(state, trace, "risk_scoring", _do,
                             f"tabular+temporal scoring for {patient['patient_id']}")
        trace.log(self.name, "evidence",
                  f"tabular={result['tabular']['model']} "
                  f"score={result['tabular']['score']}")
        return {"risk": result}

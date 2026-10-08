"""MedicationAgent: interaction + contraindication screening."""
from .base import Agent
from .medication_data import check_contraindications, check_interactions


class MedicationAgent(Agent):
    name = "MedicationAgent"

    def run(self, state, trace):
        patient = state["patient"]

        def _do():
            meds = patient.get("medications", [])
            conditions = []
            if patient.get("labels", {}).get("respiratory_flag"):
                conditions.append("asthma")
            interactions = check_interactions(meds)
            contras = check_contraindications(meds, conditions)
            flags = []
            for it in interactions:
                flags.append(
                    f"[{it['severity'].upper()}] {' + '.join(it['drugs'])}: "
                    f"{it['effect']} Advice: {it['advice']}")
            for c in contras:
                flags.append(f"[HIGH] {c['drug']} with {c['condition']}: {c['note']}")
            return {
                "medications": meds,
                "interactions": interactions,
                "contraindications": contras,
                "flags": flags,
                "n_flags": len(flags),
            }

        result = self._timed(state, trace, "medication_screening", _do,
                             f"{len(patient.get('medications', []))} medications checked")
        trace.log(self.name, "evidence",
                  f"{result['n_flags']} medication flags")
        return {"medication_review": result}

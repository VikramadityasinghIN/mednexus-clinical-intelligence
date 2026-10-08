"""ResearchAgent: RAG evidence retrieval grounded in the bundled KB."""
from .base import Agent
from models.nlp import extract_findings


class ResearchAgent(Agent):
    name = "ResearchAgent"

    def run(self, state, trace):
        patient = state["patient"]
        ctx = self.ctx

        def _do():
            retr = ctx.load_retriever()
            findings = extract_findings(patient.get("note", ""))
            terms = [f["finding"].replace("_", " ")
                     for f in findings["rule_based"] if not f["negated"]]
            risk = state.get("risk", {})
            topics = set()
            if risk.get("combined_score", 0) >= 0.4:
                topics.add("cardiac")
            if patient.get("labels", {}).get("diabetes_risk"):
                topics.add("diabetes")
            if patient.get("labels", {}).get("respiratory_flag"):
                topics.add("respiratory")
            topics.add("general")

            queries = []
            if terms:
                queries.append("patient findings: " + ", ".join(terms[:6]))
            queries.append(
                f"{patient['age']} year old {'male' if patient['sex'] == 'M' else 'female'} "
                f"cardiac risk {risk.get('combined_level', '')} "
                f"cholesterol {patient['total_chol']:.0f} bp {patient['sbp']:.0f}")
            if patient.get("medications"):
                queries.append("medication safety: " + ", ".join(patient["medications"][:4]))

            seen, evidence = set(), []
            for q in queries:
                for h in retr.rerank(q, retr.retrieve(q, k=4,
                                                      topic_boost=topics)):
                    if h["id"] in seen:
                        continue
                    seen.add(h["id"])
                    evidence.append({
                        "citation": f"[{h['id']}] {h['title']}",
                        "snippet_id": h["id"],
                        "title": h["title"],
                        "text": h["text"],
                        "topic": h["topic"],
                        "source": h["source"],
                        "score": round(h["score"], 4),
                        "query": q[:80],
                    })
                    if len(evidence) >= 8:
                        break
                if len(evidence) >= 8:
                    break
            return {"evidence": evidence, "queries": queries,
                    "retriever_backend": retr.backend}

        result = self._timed(state, trace, "evidence_retrieval", _do,
                             f"{len(state['patient'].get('medications', []))} meds, "
                             f"note findings extracted")
        trace.log(self.name, "evidence",
                  f"retrieved {len(result['evidence'])} cited snippets")
        return {"research": result}

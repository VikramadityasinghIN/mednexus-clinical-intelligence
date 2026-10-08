"""SummaryAgent: narrative report via LLM (Ollama-first) or template."""
from .base import Agent
from llm.client import available as llm_available, draft_summary


class SummaryAgent(Agent):
    name = "SummaryAgent"

    def run(self, state, trace):
        def _do():
            report = state.get("fused_report", {})
            narrative = draft_summary(report)
            return {"narrative": narrative,
                    "llm_backend": llm_available()}

        result = self._timed(state, trace, "summary_generation", _do,
                             "drafting evidence-grounded summary")
        trace.log(self.name, "evidence",
                  f"summary via {result['llm_backend']}")
        return {"summary": result}

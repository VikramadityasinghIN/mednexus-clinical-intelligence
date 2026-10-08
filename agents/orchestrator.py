"""Orchestrator: runs the 5 agents in the report's workflow order.

Tries LangGraph (StateGraph) when importable; otherwise uses the built-in
explicit state-machine orchestrator with identical agent order, state shape
and trace format. The fallback is the tested path on machines without
langgraph — see docs/ARCHITECTURE.md for the rationale.
"""
import time

from .base import Trace
from .risk_agent import RiskAgent
from .image_agent import ImageAgent
from .research_agent import ResearchAgent
from .medication_agent import MedicationAgent
from .summary_agent import SummaryAgent
from .verification import verify_report

try:
    from langgraph.graph import StateGraph, END as _LG_END
    _LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph, _LG_END = None, None
    _LANGGRAPH_AVAILABLE = False

LANGGRAPH_AVAILABLE = _LANGGRAPH_AVAILABLE

AGENT_ORDER = ["risk", "image", "research", "medication", "summary"]


def _build_agents(ctx):
    return {
        "risk": RiskAgent(ctx),
        "image": ImageAgent(ctx),
        "research": ResearchAgent(ctx),
        "medication": MedicationAgent(ctx),
        "summary": SummaryAgent(ctx),
    }


class SequentialOrchestrator:
    """Explicit state-machine orchestrator (tested fallback)."""
    backend = "sequential-state-machine"

    def __init__(self, ctx):
        self.ctx = ctx
        self.agents = _build_agents(ctx)

    def run(self, patient):
        from fusion.fusion import fuse_evidence
        trace = Trace()
        t0 = time.time()
        trace.log("Orchestrator", "start",
                  f"analyzing {patient['patient_id']} (SYNTHETIC)")
        state = {"patient": patient}

        # 1-4: evidence agents
        for key in ["risk", "image", "research", "medication"]:
            updates = self.agents[key].run(state, trace)
            state.update(updates)

        # 5: fusion before summary so the narrative has the full report
        fused = fuse_evidence(state)
        state["fused_report"] = fused
        trace.log("Fusion", "evidence_fusion",
                  f"{len(fused['conditions'])} conditions, "
                  f"{len(fused['conflicts'])} conflicts, "
                  f"{len(fused['evidence'])} citations")

        # 6: summary
        state.update(self.agents["summary"].run(state, trace))

        # 7: verification layer
        verified, issues = verify_report(state["fused_report"])
        state["fused_report"] = verified
        trace.log("Verification", "claim_check",
                  "passed" if not issues else f"{len(issues)} issues: "
                  + "; ".join(issues[:3]))

        total_ms = (time.time() - t0) * 1000
        trace.log("Orchestrator", "done",
                  f"total {total_ms:.0f} ms", {"latency_ms": total_ms})
        return {"state": state, "trace": trace.as_list(),
                "backend": self.backend}


def _try_langgraph_orchestrator(ctx):
    """Build a LangGraph StateGraph mirroring the sequential flow."""
    from fusion.fusion import fuse_evidence

    agents = _build_agents(ctx)
    trace_holder = {}

    def node(name):
        def _fn(state):
            trace = trace_holder["trace"]
            if name == "fusion":
                fused = fuse_evidence(state)
                trace.log("Fusion", "evidence_fusion", "langgraph node")
                return {"fused_report": fused}
            if name == "verification":
                verified, issues = verify_report(state["fused_report"])
                trace.log("Verification", "claim_check",
                          "passed" if not issues else f"{len(issues)} issues")
                return {"fused_report": verified}
            updates = agents[name].run(state, trace)
            return updates
        _fn.__name__ = f"node_{name}"
        return _fn

    order = ["risk", "image", "research", "medication",
             "fusion", "summary", "verification"]
    graph = StateGraph(dict)
    for name in order:
        graph.add_node(name, node(name))
    graph.set_entry_point(order[0])
    for a, b in zip(order, order[1:]):
        graph.add_edge(a, b)
    graph.add_edge(order[-1], _LG_END)
    compiled = graph.compile()

    class LangGraphOrchestrator:
        backend = "langgraph"

        def run(self, patient):
            import time as _t
            trace = Trace()
            t0 = _t.time()
            trace.log("Orchestrator", "start",
                      f"analyzing {patient['patient_id']} (SYNTHETIC) via LangGraph")
            trace_holder["trace"] = trace
            final = compiled.invoke({"patient": patient})
            total_ms = (_t.time() - t0) * 1000
            trace.log("Orchestrator", "done", f"total {total_ms:.0f} ms")
            return {"state": final, "trace": trace.as_list(),
                    "backend": self.backend}

    return LangGraphOrchestrator()


def get_orchestrator(ctx):
    """LangGraph when importable, else the tested sequential orchestrator."""
    if _LANGGRAPH_AVAILABLE:
        try:
            return _try_langgraph_orchestrator(ctx)
        except Exception:
            pass
    return SequentialOrchestrator(ctx)

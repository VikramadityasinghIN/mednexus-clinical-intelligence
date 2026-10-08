"""Agent base classes and execution tracing."""
import time


class Trace:
    """Collects a step-by-step execution log of the agent run."""

    def __init__(self):
        self.steps = []

    def log(self, agent, action, detail="", data=None):
        self.steps.append({
            "t": round(time.time(), 3),
            "agent": agent,
            "action": action,
            "detail": detail,
            "data": data if data is not None else {},
        })

    def as_list(self):
        return self.steps

    def summary(self):
        return [f"[{s['agent']}] {s['action']}: {s['detail']}" for s in self.steps]


class Agent:
    """Base class: every agent reads the shared state dict and returns
    a dict of updates merged back into state."""

    name = "base_agent"

    def __init__(self, ctx):
        self.ctx = ctx  # SystemContext: models, retriever, llm, ...

    def run(self, state, trace):
        raise NotImplementedError

    def _timed(self, state, trace, action, fn, detail=""):
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        trace.log(self.name, action,
                  f"{detail} ({dt * 1000:.0f} ms)".strip(),
                  {"latency_ms": round(dt * 1000, 1)})
        return out

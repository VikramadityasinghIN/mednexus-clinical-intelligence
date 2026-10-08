"""Train (or load cached) every model component.

Usage:
    python scripts/train_all.py [--n 1500] [--seed 42]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.context import SystemContext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ctx = SystemContext(n_patients=args.n, seed=args.seed)
    ctx.train_all(verbose=True)
    name, model, auc = ctx.best_tabular("cardiac_risk")
    print(f"Best tabular model: {name} (test AUROC={auc:.3f})")
    print(f"Temporal model: {ctx.temporal_model.name}")
    print(f"Vision backend: {ctx.vision_model.name}")
    print(f"RAG backend: {ctx.retriever.backend}")


if __name__ == "__main__":
    main()

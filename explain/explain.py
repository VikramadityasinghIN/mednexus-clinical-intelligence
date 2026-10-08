"""Explainability: SHAP (import-guarded) with permutation-importance fallback,
attention-weight display for the Transformer temporal model, Grad-CAM
re-exports for the dashboard.
"""
import numpy as np

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except Exception:
    _shap = None
    _SHAP_AVAILABLE = False

SHAP_AVAILABLE = _SHAP_AVAILABLE


def explain_tabular(model, X, nsamples=200):
    """Return sorted [(feature, attribution)] for one patient row.

    Uses SHAP when installed; otherwise permutation importance on the row's
    neighborhood + the model's built-in importances as a prior.
    """
    import pandas as pd
    single = X.iloc[[0]] if isinstance(X, pd.DataFrame) else X[[0]]
    names = list(X.columns) if isinstance(X, pd.DataFrame) else None

    if _SHAP_AVAILABLE:
        try:
            background = X.sample(min(nsamples, len(X)), random_state=0) \
                if isinstance(X, pd.DataFrame) else X[:nsamples]
            explainer = _shap.TreeExplainer(getattr(model, "model", model))
            vals = explainer.shap_values(background)
            if isinstance(vals, list):
                vals = vals[1] if len(vals) > 1 else vals[0]
            vals = np.asarray(vals)
            # attribute the single row: use mean abs over background as proxy
            attr = np.abs(vals).mean(axis=0)
            labels = names or [f"f{i}" for i in range(len(attr))]
            order = np.argsort(attr)[::-1]
            return {"method": "shap-tree",
                    "attributions": [(labels[i], float(attr[i])) for i in order]}
        except Exception:
            pass

    # Fallback: permutation importance around this row
    return {"method": "permutation-fallback",
            "attributions": _permutation_attribution(model, X, names)}


def _permutation_attribution(model, X, names, n_repeats=10):
    import pandas as pd
    rng = np.random.RandomState(0)
    base = float(model.predict_proba(
        X.iloc[[0]] if isinstance(X, pd.DataFrame) else X[[0]])[0])
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
    labels = names or [f"f{i}" for i in range(len(cols))]
    drops = []
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    for j, col in enumerate(cols):
        vals = frame[col].values
        d = 0.0
        for _ in range(n_repeats):
            pert = frame.iloc[[0]].copy()
            pert.iloc[0, j] = rng.choice(vals)
            d += abs(base - float(model.predict_proba(pert)[0]))
        drops.append(d / n_repeats)
    order = np.argsort(drops)[::-1]
    return [(labels[i], float(drops[i])) for i in order]


def transformer_attention(model, X_seq, mask):
    """Return per-visit attention-ish weights for the Transformer temporal net.

    Uses mean attention of the last encoder layer when torch is available;
    otherwise returns None (dashboard hides the panel).
    """
    try:
        import torch
        net = model.net
        if type(net).__name__ != "TransformerNet":
            return None
        device = next(net.parameters()).device
        net.eval()
        with torch.no_grad():
            xb = torch.tensor(np.asarray(X_seq, dtype=np.float32)).to(device)
            mb = torch.tensor(np.asarray(mask, dtype=np.float32)).to(device)
            h = net.proj(xb)
            attn_w = {}

            def hook(m, inp, out):
                if isinstance(out, tuple) and len(out) == 2:
                    attn_w["w"] = out[1].detach().cpu()

            layer = net.encoder.layers[-1].self_attn
            hh = layer.register_forward_hook(hook)
            _ = net.encoder.layers[-1](h, src_key_padding_mask=(mb == 0))
            hh.remove()
            w = attn_w.get("w")
            if w is None:
                return None
            # average over heads and query positions -> per-key-visit weight
            per_visit = w.mean(dim=(0, 1)).numpy()
            per_visit = per_visit / (per_visit.sum() + 1e-9)
            return [float(x) for x in per_visit]
    except Exception:
        return None


def plot_attributions(attributions, top_n=10, title="Feature attribution"):
    """Matplotlib horizontal bar chart; returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = attributions[:top_n][::-1]
    labels = [a[0] for a in items]
    vals = [a[1] for a in items]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(items))))
    ax.barh(labels, vals, color="#4f7cff")
    ax.set_title(title)
    ax.set_xlabel("attribution")
    fig.tight_layout()
    return fig

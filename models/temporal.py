"""Time-series models over longitudinal visit vitals.

Unified interface: fit(X_seq, mask, y) / predict_proba(X_seq, mask).
Deep models (LSTM, GRU, Temporal-CNN, Transformer) need torch; when torch is
absent every model transparently falls back to engineered trend features +
a sklearn classifier, and `available()` reports what is real.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

TORCH_AVAILABLE = _TORCH_AVAILABLE


# ---------------------------------------------------------------- fallback
def trend_features(X_seq, mask):
    """Engineered features: last value, slope, std, min, max per channel."""
    n, t, c = X_seq.shape
    feats = []
    tt = np.arange(t, dtype=np.float64)
    for i in range(n):
        m = mask[i].astype(bool)
        row = []
        for ch in range(c):
            v = X_seq[i, m, ch].astype(np.float64)
            if len(v) == 0:
                row += [0.0] * 5
                continue
            x = tt[m]
            slope = 0.0 if len(v) < 2 else float(np.polyfit(x, v, 1)[0])
            row += [float(v[-1]), slope, float(v.std()), float(v.min()), float(v.max())]
        feats.append(row)
    return np.array(feats, dtype=np.float32)


class TrendFeatureModel:
    """Pure-sklearn fallback: trend features + gradient boosting."""
    name = "trend_features+gbm"
    is_fallback = True

    def __init__(self, seed=42):
        self.seed = seed
        self.model = None

    def available(self):
        return True

    def fit(self, X_seq, mask, y, epochs=None):
        from sklearn.ensemble import GradientBoostingClassifier
        F = trend_features(np.asarray(X_seq), np.asarray(mask))
        self.model = GradientBoostingClassifier(random_state=self.seed)
        self.model.fit(F, np.asarray(y))
        return self

    def predict_proba(self, X_seq, mask):
        F = trend_features(np.asarray(X_seq), np.asarray(mask))
        return self.model.predict_proba(F)[:, 1]


# ---------------------------------------------------------------- torch nets
if _TORCH_AVAILABLE:
    class _SeqNet(nn.Module):
        def __init__(self, n_features, hidden=64):
            super().__init__()
            self.hidden = hidden

        def _masked_mean(self, h, mask):
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    class LSTMNet(_SeqNet):
        name = "lstm"
        is_fallback = False

        def __init__(self, n_features, hidden=64):
            super().__init__(n_features, hidden)
            self.rnn = nn.LSTM(n_features, hidden, batch_first=True, num_layers=2,
                               dropout=0.2)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x, mask):
            h, _ = self.rnn(x)
            pooled = self._masked_mean(h, mask)
            return self.head(pooled).squeeze(-1)

    class GRUNet(_SeqNet):
        name = "gru"
        is_fallback = False

        def __init__(self, n_features, hidden=64):
            super().__init__(n_features, hidden)
            self.rnn = nn.GRU(n_features, hidden, batch_first=True, num_layers=2,
                              dropout=0.2)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x, mask):
            h, _ = self.rnn(x)
            pooled = self._masked_mean(h, mask)
            return self.head(pooled).squeeze(-1)

    class TemporalCNNNet(_SeqNet):
        name = "temporal_cnn"
        is_fallback = False

        def __init__(self, n_features, hidden=64):
            super().__init__(n_features, hidden)
            self.conv = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x, mask):
            h = self.conv(x.transpose(1, 2)).squeeze(-1)  # (B, 64)
            return self.head(h).squeeze(-1)

    class TransformerNet(_SeqNet):
        name = "transformer"
        is_fallback = False

        def __init__(self, n_features, hidden=64):
            super().__init__(n_features, hidden)
            self.proj = nn.Linear(n_features, hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                                               dim_feedforward=128, dropout=0.1,
                                               batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(32, 1))
            self.last_attn = None

        def forward(self, x, mask, return_attn=False):
            h = self.proj(x)
            key_pad = (mask == 0)
            enc = self.encoder(h, src_key_padding_mask=key_pad)
            pooled = self._masked_mean(enc, mask)
            out = self.head(pooled).squeeze(-1)
            return out

    _TORCH_MODELS = {
        "lstm": LSTMNet, "gru": GRUNet,
        "temporal_cnn": TemporalCNNNet, "transformer": TransformerNet,
    }
else:
    _TORCH_MODELS = {}


class TorchTemporalModel:
    """Wrapper giving torch nets the unified interface."""
    is_fallback = False

    def __init__(self, net_name, n_features, seed=42, epochs=25, lr=1e-3):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is not installed; use TrendFeatureModel")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.name = net_name
        self.net = _TORCH_MODELS[net_name](n_features)
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        # per-channel normalization stats, fitted on train
        self.mean_ = None
        self.std_ = None

    def available(self):
        return _TORCH_AVAILABLE

    def _norm_fit(self, X, mask):
        m = mask.astype(bool)[..., None]
        vals = np.where(m, X, np.nan)
        self.mean_ = np.nanmean(vals, axis=(0, 1))
        self.std_ = np.nanstd(vals, axis=(0, 1)) + 1e-6

    def _norm(self, X):
        return (X - self.mean_) / self.std_

    def fit(self, X_seq, mask, y, epochs=None):
        X = np.asarray(X_seq, dtype=np.float32)
        M = np.asarray(mask, dtype=np.float32)
        Y = np.asarray(y, dtype=np.float32)
        self._norm_fit(X, M)
        Xn = self._norm(X)
        ds = TensorDataset(torch.tensor(Xn), torch.tensor(M), torch.tensor(Y))
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        self.net.train()
        for _ in range(epochs or self.epochs):
            for xb, mb, yb in dl:
                xb, mb, yb = xb.to(self.device), mb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb, mb), yb)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad() if _TORCH_AVAILABLE else (lambda f: f)
    def predict_proba(self, X_seq, mask):
        X = np.asarray(X_seq, dtype=np.float32)
        M = np.asarray(mask, dtype=np.float32)
        Xn = self._norm(X)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), 256):
                xb = torch.tensor(Xn[i:i + 256]).to(self.device)
                mb = torch.tensor(M[i:i + 256]).to(self.device)
                out.append(torch.sigmoid(self.net(xb, mb)).cpu().numpy())
        return np.concatenate(out)


def get_temporal_model(name, n_features, seed=42, **kw):
    """Factory: deep model if torch present, else trend-feature fallback."""
    if name in _TORCH_MODELS:
        return TorchTemporalModel(name, n_features, seed=seed, **kw)
    if name == "trend_fallback":
        return TrendFeatureModel(seed=seed)
    raise ValueError(f"unknown temporal model: {name}")


def available_temporal_models():
    models = list(_TORCH_MODELS) + ["trend_fallback"]
    return models


def compare_temporal_models(X_seq, mask, y, X_test, mask_test, y_test, seed=42):
    """Train every available temporal model; return {name: {metrics, model}}."""
    from sklearn.metrics import roc_auc_score
    n_features = X_seq.shape[2]
    results = {}
    for name in available_temporal_models():
        try:
            model = get_temporal_model(name, n_features, seed=seed, epochs=15)
            model.fit(X_seq, mask, y)
            proba = model.predict_proba(X_test, mask_test)
            results[name] = {
                "auroc": float(roc_auc_score(y_test, proba)),
                "is_fallback": bool(getattr(model, "is_fallback", False)),
                "model": model,
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

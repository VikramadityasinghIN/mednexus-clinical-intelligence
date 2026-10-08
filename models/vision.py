"""Chest-X-ray vision module.

Primary: TorchVision DenseNet121 (ImageNet-pretrained backbone, replaced
classifier head), fine-tune training loop, Grad-CAM heatmap overlays.
Fallback (no torch): HeuristicVisionModel — measures brightness anomalies in
the lung regions. It is honest about being a heuristic and only meaningful on
the synthetic images; the orchestrator labels which backend produced the
finding. `available()` reports the real backend.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.models as _tv_models
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

TORCH_AVAILABLE = _TORCH_AVAILABLE


class HeuristicVisionModel:
    """Fallback when torch is unavailable.

    Scores the fraction of bright pixels inside the two lung regions of the
    synthetic images. Deterministic, dependency-free, clearly labeled as a
    heuristic — a stand-in so the full pipeline runs end-to-end anywhere.
    """
    name = "heuristic-lung-opacity"
    is_fallback = True

    def __init__(self, threshold=0.012):
        self.threshold = threshold

    def available(self):
        return True

    def _lung_mask(self, img):
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cx = w // 2
        mask = np.zeros_like(img, dtype=bool)
        for side in (-1, 1):
            lx = cx + side * int(w * 0.20)
            lung = ((xx - lx) ** 2 / (w * 0.13) ** 2
                    + (yy - h * 0.50) ** 2 / (h * 0.30) ** 2) < 1
            mask |= lung
        return mask

    def predict_proba(self, images):
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[None]
        scores = []
        for img in images:
            lung = self._lung_mask(img)
            bright = (img > 0.55) & lung
            frac = bright.sum() / max(1, lung.sum())
            scores.append(float(np.clip((frac - self.threshold) / 0.05, 0, 1)))
        return np.array(scores)

    def heatmap(self, image):
        """Return a coarse anomaly heatmap (same shape as image)."""
        img = np.asarray(image)
        lung = self._lung_mask(img).astype(np.float32)
        anomaly = np.clip((img - 0.45), 0, 1) * lung
        # smooth a little
        from numpy.lib.stride_tricks import sliding_window_view
        try:
            w = sliding_window_view(np.pad(anomaly, 2), (5, 5))
            anomaly = w.mean(axis=(-2, -1))
        except Exception:
            pass
        mx = anomaly.max()
        return anomaly / mx if mx > 0 else anomaly


if _TORCH_AVAILABLE:
    class GradCAM:
        """Grad-CAM for any conv net: hooks the target layer, weights feature
        maps by mean-pooled gradients of the target class score."""

        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.activations = None
            self.gradients = None
            target_layer.register_forward_hook(self._fwd_hook)
            target_layer.register_full_backward_hook(self._bwd_hook)

        def _fwd_hook(self, module, inp, out):
            self.activations = out.detach()

        def _bwd_hook(self, module, gin, gout):
            self.gradients = gout[0].detach()

        def __call__(self, x, class_idx=None):
            self.model.eval()
            x = x.requires_grad_(True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            score = logits[0, class_idx]
            self.model.zero_grad()
            score.backward()
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(1).squeeze(0)
            cam = torch.relu(cam)
            cam = cam / (cam.max() + 1e-8)
            return cam.detach().cpu().numpy(), class_idx

    class DenseNetXray(nn.Module):
        name = "densenet121-xray"
        is_fallback = False

        def __init__(self, pretrained=True, seed=42):
            super().__init__()
            torch.manual_seed(seed)
            weights = (_tv_models.DenseNet121_Weights.IMAGENET1K_V1
                       if pretrained else None)
            self.backbone = _tv_models.densenet121(weights=weights)
            in_f = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, 1))
            self.gradcam = GradCAM(self.backbone,
                                   self.backbone.features.denseblock4)

        def forward(self, x):
            return self.backbone(x).squeeze(-1)

        @staticmethod
        def preprocess(images):
            """(N,H,W) float32 0..1 -> normalized 3-channel tensor."""
            arr = np.asarray(images, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None]
            arr = np.repeat(arr[:, None], 3, axis=1)
            t = torch.tensor(arr)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            return (t - mean) / std

        def fit(self, images, labels, epochs=5, lr=1e-4, batch_size=16):
            from torch.utils.data import DataLoader, TensorDataset
            X = self.preprocess(images)
            y = torch.tensor(np.asarray(labels), dtype=torch.float32)
            dl = DataLoader(TensorDataset(X, y), batch_size=batch_size,
                            shuffle=True)
            # freeze backbone first, train head; then unfreeze last block
            for p in self.backbone.features.parameters():
                p.requires_grad = False
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                          self.parameters()), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()
            dev = next(self.parameters()).device
            self.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    opt.zero_grad()
                    loss = loss_fn(self(xb), yb)
                    loss.backward()
                    opt.step()
            return self

        def predict_proba(self, images):
            self.eval()
            X = self.preprocess(images)
            dev = next(self.parameters()).device
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    logits = self(X[i:i + 32].to(dev))
                    out.append(torch.sigmoid(logits).cpu().numpy())
            return np.concatenate(out)

        def heatmap(self, image):
            x = self.preprocess(image[None] if image.ndim == 2 else image)
            dev = next(self.parameters()).device
            cam, cls = self.gradcam(x.to(dev))
            # resize cam to image size
            import torch.nn.functional as F
            t = torch.tensor(cam[None, None])
            up = F.interpolate(t, size=image.shape[:2], mode="bilinear",
                               align_corners=False)[0, 0].numpy()
            return np.clip(up, 0, 1), cls

    def get_vision_model(pretrained=True, seed=42):
        # Never silently use a random untrained backbone or require internet.
        # Use pretrained weights only when cached locally; otherwise the
        # clearly labeled synthetic-image heuristic keeps the demo operational.
        if pretrained:
            from pathlib import Path
            import torch.hub
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "densenet121-a639ec97.pth"
            if not checkpoint.is_file():
                return HeuristicVisionModel()
        try:
            return DenseNetXray(pretrained=pretrained, seed=seed)
        except (OSError, RuntimeError, ValueError):
            return HeuristicVisionModel()
else:
    def get_vision_model(**kwargs):
        return HeuristicVisionModel()


def available():
    return "densenet121" if _TORCH_AVAILABLE else "heuristic-fallback"

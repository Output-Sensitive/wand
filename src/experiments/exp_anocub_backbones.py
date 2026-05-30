"""Reproduce the AnoCUB dataset + the supplement's embedding ablations.

AnoCUB is derived from CUB-200-2011 (not a standard download), so this
script regenerates it and the two supplement tables:

  1. Whitening ablation  (Table: ResNet-18 embedding, whiten ON/OFF,
     raw vs z-scored).
  2. Lightweight-backbone comparison (z-scored, no whitening):
     ResNet-18, EfficientNet-B0, MobileNetV3-small.

It also (re)writes datasets/cub/anocub_task.npz via exp_e7_anocub.

Usage:  python src/exp_anocub_backbones.py
Requires: torch, torchvision, the CUB archive under datasets/cub/
(downloaded automatically by exp_e7_anocub if absent).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from sklearn.metrics import roc_auc_score
from core.explain import WANDExplainer            # noqa: E402
from experiments import exp_e7_anocub as A7                       # noqa: E402

IMNORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def build_task():
    """Build AnoCUB (writes anocub_task.npz) and return (idx, y, paths, base)."""
    base = A7.ensure_extracted()
    _A, paths, cls, class_names, _pl, attr_names, _parts = A7.build(base)
    idx, y, _, _ = A7.select_task(cls, class_names)
    return base, paths, idx, y


def embed(base, paths, idx, make_model):
    import torchvision.transforms as T
    from PIL import Image
    resize = T.Compose([T.Resize(256), T.CenterCrop(224)])
    to_tensor, norm = T.ToTensor(), T.Normalize(*IMNORM)
    m = make_model(); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    outs = []
    with torch.no_grad():
        gids = list(idx)
        for k in range(0, len(gids), 64):
            xs = torch.stack([norm(to_tensor(resize(
                Image.open(str(base / "images" / paths[g])).convert("RGB"))))
                for g in gids[k:k + 64]])
            outs.append(m(xs))
    return torch.cat(outs).double().numpy()


def auc(E, y, whiten, zscore):
    X = (E - E.mean(0)) / (E.std(0) + 1e-9) if zscore else E
    return roc_auc_score(y, WANDExplainer(K=1024, seed=0, whiten=whiten).fit(X).score(X))


def backbones():
    from torchvision import models as M
    def r18():
        m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT); m.fc = torch.nn.Identity(); return m
    def eff():
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.DEFAULT); m.classifier = torch.nn.Identity(); return m
    def mob():
        m = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.DEFAULT); m.classifier = torch.nn.Identity(); return m
    return [("ResNet-18", r18), ("EfficientNet-B0", eff), ("MobileNetV3-small", mob)]


def main():
    base, paths, idx, y = build_task()
    print(f"AnoCUB: {len(idx)} images, {int(y.sum())} anomalies "
          f"(task at datasets/cub/anocub_task.npz)\n")

    bb = backbones()
    embs = {name: embed(base, paths, idx, mk) for name, mk in bb}

    print("== Whitening ablation (ResNet-18 embedding) ==")
    E = embs["ResNet-18"]
    print(f"{'':10s} whiten=ON  whiten=OFF")
    print(f"{'raw':10s} {auc(E,y,True,False):.3f}     {auc(E,y,False,False):.3f}")
    print(f"{'z-scored':10s} {auc(E,y,True,True):.3f}     {auc(E,y,False,True):.3f}")

    print("\n== Backbone comparison (z-scored, no whitening) ==")
    print(f"{'backbone':18s} dim   AUC")
    for name, _ in bb:
        E = embs[name]
        print(f"{name:18s} {E.shape[1]:4d}  {auc(E,y,False,True):.3f}")


if __name__ == "__main__":
    main()

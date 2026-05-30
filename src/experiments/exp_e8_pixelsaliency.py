"""E8 -- Pixel-level WAND explanation on raw images.

WAND's score is differentiable in its input features.  If those features
come from a differentiable encoder phi (here a frozen, pre-trained CNN),
then d(WAND score)/d(pixels) is just autograd through phi -- a saliency
map, with no training.  We embed the AnoCUB images with a frozen
ResNet, fit WAND on the embeddings, and back-propagate the anomaly score
to the pixels (SmoothGrad) to highlight what makes a flagged bird
anomalous.  This realises the pixel-level explanation the main paper
leaves as a remark.

Outputs:
  figures/anocub_pixsal.pdf   -- flagged birds + pixel saliency
  prints detection AUC on the embedding space.

Usage:  python src/exp_e8_pixelsaliency.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from core.explain import WANDExplainer            # noqa: E402
from experiments import exp_e7_anocub as A7                       # noqa: E402

IMNORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def encoder():
    import torchvision as tv
    from torchvision.models import resnet18, ResNet18_Weights
    w = ResNet18_Weights.DEFAULT
    m = resnet18(weights=w)
    m.fc = torch.nn.Identity()                   # -> 512-d avgpool features
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, w.transforms()


def main():
    base = A7.ensure_extracted()
    _A, paths, cls, class_names, part_locs, attr_names, parts = A7.build(base)
    idx, y, in_idx, an_idx = A7.select_task(cls, class_names)

    import torchvision.transforms as T
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    enc, prep = encoder()
    # transform for embedding (ImageNet preprocessing) and a raw 224 view
    resize = T.Compose([T.Resize(256), T.CenterCrop(224)])
    to_tensor = T.ToTensor()
    norm = T.Normalize(*IMNORM)

    def load_img(gid):
        im = Image.open(str(base / "images" / paths[gid])).convert("RGB")
        return resize(im)

    # ---- embed all selected images (frozen encoder); cache to disk ----
    cache = A7.CUB / "anocub_resnet18.npy"
    if cache.exists():
        E = np.load(cache)
        print(f"loaded cached embeddings {E.shape}")
    else:
        print(f"embedding {len(idx)} images with frozen ResNet18 ...", flush=True)
        embs = []
        with torch.no_grad():
            gids = list(idx)
            for k in range(0, len(gids), 32):
                chunk = gids[k:k + 32]
                xs = torch.stack([norm(to_tensor(load_img(g))) for g in chunk])
                embs.append(enc(xs))
            E = torch.cat(embs).double().numpy()
        np.save(cache, E)
    # z-score the embeddings and fit WITHOUT Mahalanobis whitening: on the
    # contaminated 512-d covariance, whitening suppresses the very anomaly
    # directions, dropping AUC to ~0.88; z-score + no-whiten gives ~0.99.
    emb_mu, emb_sd = E.mean(0, keepdims=True), E.std(0, keepdims=True) + 1e-9
    Ez = (E - emb_mu) / emb_sd
    mu_t = torch.tensor(emb_mu, dtype=torch.float64)
    sd_t = torch.tensor(emb_sd, dtype=torch.float64)
    expl = WANDExplainer(K=1024, seed=0, whiten=False).fit(Ez)
    s = expl.score(Ez)
    auc = roc_auc_score(y, s)
    print(f"embedding-space detection AUC = {auc:.3f}")

    order = np.argsort(s)[::-1]
    # top-3 distinct flagged *true-anomaly* species (cross-family birds)
    seen, show = set(), []
    for loc in order:
        if y[loc] != 1:
            continue
        c = cls[idx[loc]]
        if c in seen:
            continue
        seen.add(c); show.append(loc)
        if len(show) == 3:
            break

    def saliency(gid, n_smooth=15, sigma=0.12):
        """SmoothGrad |d score/d pixels| through the frozen encoder."""
        base_t = norm(to_tensor(load_img(gid)))            # (3,224,224)
        rng = torch.Generator().manual_seed(0)
        acc = torch.zeros(224, 224)
        rng_std = sigma * (base_t.max() - base_t.min())
        for _ in range(n_smooth):
            noise = torch.randn(base_t.shape, generator=rng) * rng_std
            xi = (base_t + noise).unsqueeze(0).requires_grad_(True)
            sc = expl.score_torch((enc(xi).double() - mu_t) / sd_t)
            g, = torch.autograd.grad(sc.sum(), xi)
            acc += g.abs().squeeze(0).sum(0)
        sal = acc.numpy()
        from scipy.ndimage import gaussian_filter
        sal = gaussian_filter(sal, sigma=6)
        return sal - sal.min()                  # raw (shared scale set later)

    # ---------------- figure: 3 outliers + 1 inlier (shared scale) -------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})

    def pct(loc):
        return 100.0 * float((s < s[loc]).mean())

    in_locs = np.where(y == 0)[0]
    inlier = in_locs[int(np.argmin(s[in_locs]))]     # most-normal inlier
    panels = [(loc, False) for loc in show] + [(inlier, True)]

    maps = [saliency(idx[loc]) for loc, _ in panels]
    gmax = max(float(m.max()) for m in maps) + 1e-9   # shared colour scale

    import textwrap
    fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.6))
    im = None
    for j, ((loc, is_in), sal) in enumerate(zip(panels, maps)):
        gid = idx[loc]
        ax[j].imshow(np.asarray(load_img(gid)))
        im = ax[j].imshow(sal / gmax, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax[j].axis("off")
        sp = class_names[cls[gid]].split(".", 1)[-1].replace("_", " ")
        sp = "\n".join(textwrap.wrap(sp, 18)) or sp
        tag = "inlier" if is_in else "flagged"
        col = "#1a7d1a" if is_in else "#b30000"
        ax[j].set_title(f"{sp}\n{pct(loc):.1f}th pct, {tag}", fontsize=10, color=col)
        print(f"  saliency: {sp} ({pct(loc):.1f}th pct, {'inlier' if is_in else 'outlier'}), "
              f"peak={sal.max()/gmax:.2f}")
    fig.subplots_adjust(top=0.82, wspace=0.05)
    cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    cb.set_label("saliency (shared scale)", fontsize=10); cb.ax.tick_params(labelsize=9)
    out = ROOT / "figures" / "anocub_pixsal.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

"""E9 -- AnoCUB explanation gallery for the supplement: concept-level
(which attributes) vs pixel-level (where), with an honest look at where
each mode is weak.

Two rows, three panels each:
  Row 1  Concept-level part heatmaps (witness attribution -> CUB keypoints):
         pelican (ok), hummingbird (ok -- detected at the 99th pct), and a
         Mallard whose TOP concept is the *global* "duck-like" shape, which
         has no body-part keypoint -> the part heatmap cannot localise it
         (a localisation miss, not a detection miss).
  Row 2  Pixel-level saliency (d score/d pixels through a frozen ResNet-18):
         pelican (ok), mallard (ok), and the Anna Hummingbird that is the
         embedding near-miss (rank ~131, 89th pct): small hummingbirds embed
         close to the sparrow inliers, so its saliency is weak/diffuse.

The point (fair, no over-claim): the two explanation modes have
complementary blind spots -- the hummingbird the embedding nearly misses
is cleanly handled in concept space, and the Mallard whose dominant
concept is global is cleanly localised in pixel space.

Output: figures/anocub_supp_gallery.pdf
Usage:  python src/exp_e9_anocub_supp.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch                                              # noqa: E402
from experiments import exp_e7_anocub as A7                                # noqa: E402
from experiments import exp_e8_pixelsaliency as A8                         # noqa: E402
from core.explain import WANDExplainer                      # noqa: E402


def species_of(class_names, cls, idx, loc):
    return class_names[cls[idx[loc]]].split(".", 1)[-1].replace("_", " ")


def main():
    # ============================= concept side =========================
    base = A7.ensure_extracted()
    _A, paths, cls, class_names, part_locs, attr_names, parts = A7.build(base)
    cont = np.loadtxt(base / "attributes" /
                      "class_attribute_labels_continuous.txt") / 100.0
    rng = np.random.default_rng(0)
    feats = cont[cls - 1] + rng.normal(0, 0.02, (len(cls), A7.N_ATTR))
    idx, y, in_idx, an_idx = A7.select_task(cls, class_names)
    X = feats[idx]
    keep = X.std(0) > 1e-9
    Xk, ki = X[:, keep], np.where(keep)[0]
    cexpl = WANDExplainer(K=1024, seed=0, whiten=False).fit(Xk)
    cs = cexpl.score(Xk)
    a2p = A7.attr_part_map(attr_names)

    def cpct(loc):
        return 100.0 * float((cs < cs[loc]).mean())

    def sp(loc):
        return species_of(class_names, cls, idx, loc)

    def cattr(loc):
        aw = np.zeros(A7.N_ATTR)
        aw[ki] = cexpl.witness_attribution(Xk[loc:loc + 1])[0]
        ps = {p: 0.0 for p in parts.values()}
        for j, v in enumerate(aw):
            for pn in a2p.get(j, []):
                ps[pn] += v
        glob = sum(v for j, v in enumerate(aw) if not a2p.get(j, []))
        return aw, ps, glob / (aw.sum() + 1e-12)

    anom = np.where(y == 1)[0]
    by_score = sorted(anom, key=lambda l: -cs[l])

    def first(kw, pred=None):
        for loc in by_score:
            if kw in sp(loc) and (pred is None or pred(loc)):
                return loc
        return None

    c_pelican = first("Pelican")
    c_humming = first("Hummingbird")
    # the Mallard whose single most-responsible concept is global (no keypoint)
    c_mallard_glob = first("Mallard",
                           lambda l: not a2p.get(int(np.argmax(_cattr_aw(cexpl, Xk, ki, l))), []))
    if c_mallard_glob is None:
        c_mallard_glob = first("Mallard")
    concept_panels = [(c_pelican, "ok"), (c_humming, "ok"),
                      (c_mallard_glob, "miss")]

    # bounding boxes for the bbox-cropped view
    bbox = {}
    for line in (base / "bounding_boxes.txt").read_text().splitlines():
        t = line.split()
        bbox[int(t[0])] = [float(v) for v in t[1:5]]

    import matplotlib.image as mpimg

    def load_crop(gid):
        im = mpimg.imread(str(base / "images" / paths[gid]))
        Hh, Ww = im.shape[:2]
        bx, by, bw, bh = bbox[gid + 1]
        pad = 0.15
        cx0, cy0 = max(0, int(bx - pad * bw)), max(0, int(by - pad * bh))
        cx1 = min(Ww, int(bx + bw + pad * bw))
        cy1 = min(Hh, int(by + bh + pad * bh))
        return im[cy0:cy1, cx0:cx1], cx0, cy0

    # ============================= pixel side ===========================
    E = np.load(A7.CUB / "anocub_resnet18.npy")
    emb_mu, emb_sd = E.mean(0, keepdims=True), E.std(0, keepdims=True) + 1e-9
    Ez = (E - emb_mu) / emb_sd
    mu_t = torch.tensor(emb_mu, dtype=torch.float64)
    sd_t = torch.tensor(emb_sd, dtype=torch.float64)
    pexpl = WANDExplainer(K=1024, seed=0, whiten=False).fit(Ez)
    psco = pexpl.score(Ez)

    def ppct(loc):
        return 100.0 * float((psco < psco[loc]).mean())

    import torchvision.transforms as T
    from PIL import Image
    enc, _ = A8.encoder()
    resize = T.Compose([T.Resize(256), T.CenterCrop(224)])
    to_tensor = T.ToTensor()
    norm = T.Normalize(*A8.IMNORM)

    def load_img(gid):
        im = Image.open(str(base / "images" / paths[gid])).convert("RGB")
        return resize(im)

    def saliency(gid, n_smooth=15, sigma=0.12):
        base_t = norm(to_tensor(load_img(gid)))
        g = torch.Generator().manual_seed(0)
        acc = torch.zeros(224, 224)
        rng_std = sigma * (base_t.max() - base_t.min())
        for _ in range(n_smooth):
            noise = torch.randn(base_t.shape, generator=g) * rng_std
            xi = (base_t + noise).unsqueeze(0).requires_grad_(True)
            sc = pexpl.score_torch((enc(xi).double() - mu_t) / sd_t)
            grad, = torch.autograd.grad(sc.sum(), xi)
            acc += grad.abs().squeeze(0).sum(0)
        from scipy.ndimage import gaussian_filter
        sal = gaussian_filter(acc.numpy(), sigma=6)
        return sal - sal.min()

    p_by_score = sorted(anom, key=lambda l: -psco[l])

    def pfirst(kw):
        for loc in p_by_score:
            if kw in sp(loc):
                return loc
        return None

    p_pelican = pfirst("Pelican")
    p_mallard = pfirst("Mallard")
    p_nearmiss = min(anom, key=lambda l: psco[l])     # lowest-scored anomaly
    pixel_panels = [(p_pelican, "ok"), (p_mallard, "ok"),
                    (p_nearmiss, "miss")]

    # ============================= figure ===============================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    cmap = plt.cm.autumn_r

    fig, ax = plt.subplots(2, 3, figsize=(8.6, 4.5))

    # ---- row 1: concept-level part heatmaps ----
    sc = None
    for col, (loc, kind) in enumerate(concept_panels):
        gid = idx[loc]
        im, ox, oy = load_crop(gid)
        a = ax[0, col]
        a.imshow(im)
        a.axis("off")
        aw, ps, gfrac = cattr(loc)
        smax = max(ps.values()) + 1e-9
        for pid, pname in parts.items():
            x, yy, vis = part_locs[gid][pid]
            if vis < 1:
                continue
            x, yy = x - ox, yy - oy
            w = ps.get(pname, 0.0) / smax
            if w <= 0.05:
                continue
            sc = a.scatter([x], [yy], s=60 + 520 * w, c=[w], cmap=cmap,
                           vmin=0, vmax=1, alpha=0.85, edgecolor="k",
                           linewidth=0.6, zorder=3)
        j_top = int(np.argmax(aw))
        top = attr_names[j_top + 1].replace("has_", "").replace("::", ": ")
        top_parts = a2p.get(j_top, [])
        if kind == "miss":
            ttl = (f"{sp(loc)}\n{cpct(loc):.0f}th pct — MISS\n"
                   f"global shape, no keypoint")
            col_t = "#b30000"
        else:
            kp = top_parts[0] if top_parts else "body"
            ttl = (f"{sp(loc)}\n{cpct(loc):.0f}th pct → {kp}\n{top}")
            col_t = "#1a7d1a"
        a.set_title(ttl, fontsize=8.5, color=col_t)

    # ---- row 2: pixel-level saliency ----
    maps = [saliency(idx[loc]) for loc, _ in pixel_panels]
    gmax = max(float(m.max()) for m in maps) + 1e-9
    pim = None
    for col, ((loc, kind), sal) in enumerate(zip(pixel_panels, maps)):
        gid = idx[loc]
        a = ax[1, col]
        a.imshow(np.asarray(load_img(gid)))
        pim = a.imshow(sal / gmax, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        a.axis("off")
        peak = sal.max() / gmax
        if kind == "miss":
            ttl = (f"{sp(loc)}\n{ppct(loc):.0f}th pct — NEAR-MISS\n"
                   f"weak saliency (peak {peak:.2f})")
            col_t = "#b30000"
        else:
            ttl = (f"{sp(loc)}\n{ppct(loc):.0f}th pct\n"
                   f"saliency peak {peak:.2f}")
            col_t = "#1a7d1a"
        a.set_title(ttl, fontsize=8.5, color=col_t)

    # row labels on the far left
    ax[0, 0].text(-0.13, 0.5, "Concept-level\n(which attribute)",
                  transform=ax[0, 0].transAxes, rotation=90, va="center",
                  ha="center", fontsize=10.5, fontweight="bold")
    ax[1, 0].text(-0.13, 0.5, "Pixel-level\n(where in image)",
                  transform=ax[1, 0].transAxes, rotation=90, va="center",
                  ha="center", fontsize=10.5, fontweight="bold")

    # lay out the 2x3 image grid first, leaving a clear right margin, then
    # drop each colorbar into that reserved margin (so it never sits on an
    # image, which a post-hoc subplots_adjust would otherwise cause).
    fig.subplots_adjust(left=0.08, right=0.88, top=0.84, bottom=0.03,
                        wspace=0.08, hspace=0.34)
    for mapp, lab in [(sc, "part attribution"), (pim, "saliency (shared)")]:
        if mapp is None:
            continue
        row = 0 if lab.startswith("part") else 1
        p = ax[row, 2].get_position()
        cax = fig.add_axes([0.895, p.y0 + 0.12 * p.height,
                            0.014, 0.76 * p.height])
        cb = fig.colorbar(mapp, cax=cax)
        cb.set_label(lab, fontsize=9.5)
        cb.ax.tick_params(labelsize=8.5)
    out = ROOT / "figures" / "anocub_supp_gallery.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"saved -> {out}")
    print("concept panels:", [(sp(l), round(cpct(l), 1), k) for l, k in concept_panels])
    print("pixel panels  :", [(sp(l), round(ppct(l), 1), k) for l, k in pixel_panels])


def _cattr_aw(expl, Xk, ki, loc):
    aw = np.zeros(A7.N_ATTR)
    aw[ki] = expl.witness_attribution(Xk[loc:loc + 1])[0]
    return aw


if __name__ == "__main__":
    main()

"""E7 -- AnoCUB: an image anomaly-detection case study with a
part-grounded explanation.

We turn CUB-200-2011 (birds) into an anomaly-detection task in the
interpretable concept space the dataset already provides: each image is
represented by its 312 named binary attributes (bill shape, wing colour,
size, ...).  Inliers are a set of visually similar species (sparrows);
anomalies are a few birds from very different families (pelican,
frigatebird, mallard, hummingbird).  WAND flags the odd birds and its
witness attribution names the responsible attributes; we then map those
attributes to CUB's 15 body-part keypoints and draw the explanation as a
heatmap directly on the bird photo.

Outputs:
  datasets/cub/anocub.npz                 -- cached attribute matrix + meta
  figures/anocub_casestudy.pdf      -- part-grounded explanation
  prints detection AUC and the localized parts.

Usage:  python src/exp_e7_anocub.py
"""
from __future__ import annotations
import sys, tarfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from core.explain import WANDExplainer            # noqa: E402

CUB = ROOT / "datasets" / "cub"
N_ATTR = 312

# Attribute-prefix -> CUB part keypoint(s).  Global attributes (size,
# shape, primary_color) have no single part and are dropped from the
# part heatmap (still used for detection and the named-attribute panel).
PREFIX_TO_PARTS = {
    "has_bill": ["beak"], "has_beak": ["beak"],
    "has_wing": ["left wing", "right wing"],
    "has_belly": ["belly"], "has_underparts": ["belly"],
    "has_breast": ["breast"],
    "has_back": ["back"], "has_upperparts": ["back"],
    "has_tail": ["tail"], "has_upper_tail": ["tail"], "has_under_tail": ["tail"],
    "has_throat": ["throat"],
    "has_crown": ["crown"], "has_head": ["crown"],
    "has_forehead": ["forehead"],
    "has_nape": ["nape"],
    "has_eye": ["left eye", "right eye"],
    "has_leg": ["left leg", "right leg"],
}


def _find(root: Path, name: str) -> Path | None:
    for c in [root / name, root / "CUB_200_2011" / name,
              root / "attributes" / name, root / "CUB_200_2011" / "attributes" / name]:
        if c.exists():
            return c
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def ensure_extracted():
    base = CUB / "CUB_200_2011"
    if base.exists() and (base / "images.txt").exists():
        return base
    tgz = CUB / "CUB_200_2011.tgz"
    if not tgz.exists():
        raise FileNotFoundError(f"{tgz} not found; download it first.")
    print("extracting CUB tarball ...", flush=True)
    with tarfile.open(tgz) as t:
        t.extractall(CUB)
    return base


def build(base: Path):
    cache = CUB / "anocub.npz"
    # ---- attribute names + part list ----
    attr_file = _find(CUB, "attributes.txt") or _find(base, "attributes.txt")
    attr_names = {}
    for line in attr_file.read_text().splitlines():
        i, nm = line.split(" ", 1)
        attr_names[int(i)] = nm.strip()
    parts = {}
    for line in (base / "parts" / "parts.txt").read_text().splitlines():
        toks = line.split(" ", 1)
        parts[int(toks[0])] = toks[1].strip()

    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return (d["A"], d["img_paths"], d["img_class"], d["class_names"].item(),
                d["part_locs"], attr_names, parts)

    # ---- image id -> path, class ----
    img_paths, img_class = {}, {}
    for line in (base / "images.txt").read_text().splitlines():
        i, p = line.split(" ", 1); img_paths[int(i)] = p.strip()
    for line in (base / "image_class_labels.txt").read_text().splitlines():
        i, c = line.split(); img_class[int(i)] = int(c)
    class_names = {}
    for line in (base / "classes.txt").read_text().splitlines():
        c, nm = line.split(" ", 1); class_names[int(c)] = nm.strip()
    n_img = len(img_paths)

    # ---- attribute matrix (n_img x 312), is_present ----
    A = np.zeros((n_img, N_ATTR), dtype=np.float32)
    ial = _find(base, "image_attribute_labels.txt")
    with open(ial) as f:
        for line in f:
            t = line.split()
            if len(t) < 3:
                continue
            iid, aid, pres = int(t[0]), int(t[1]), int(t[2])
            A[iid - 1, aid - 1] = pres

    # ---- part locations: (n_img x 15 x 3) x,y,visible ----
    part_locs = np.zeros((n_img, 16, 3), dtype=np.float32)   # 1-indexed parts
    for line in (base / "parts" / "part_locs.txt").read_text().splitlines():
        t = line.split()
        iid, pid, x, y, vis = int(t[0]), int(t[1]), float(t[2]), float(t[3]), float(t[4])
        part_locs[iid - 1, pid] = (x, y, vis)

    ids = np.array(sorted(img_paths))
    paths = np.array([img_paths[i] for i in ids])
    cls = np.array([img_class[i] for i in ids])
    np.savez_compressed(cache, A=A, img_paths=paths, img_class=cls,
                        class_names=np.array(class_names, dtype=object),
                        part_locs=part_locs)
    print(f"cached {cache}", flush=True)
    return A, paths, cls, class_names, part_locs, attr_names, parts


def select_task(cls, class_names, inlier_kw=("Sparrow",),
                anom_kw=("Pelican", "Frigatebird", "Mallard", "Hummingbird"),
                n_anom=15, seed=0):
    rng = np.random.default_rng(seed)
    name = {c: class_names[c] for c in class_names}
    is_in = np.array([any(k in name[c] for k in inlier_kw) for c in cls])
    is_an = np.array([any(k in name[c] for k in anom_kw) for c in cls])
    in_idx = np.where(is_in)[0]
    an_pool = np.where(is_an)[0]
    an_idx = rng.choice(an_pool, min(n_anom, len(an_pool)), replace=False)
    idx = np.concatenate([in_idx, an_idx])
    y = np.r_[np.zeros(len(in_idx)), np.ones(len(an_idx))].astype(int)
    return idx, y, in_idx, an_idx


def attr_part_map(attr_names):
    """attr index (0-based) -> list of part names (or [] if global)."""
    m = {}
    for aid, nm in attr_names.items():
        pre = nm.split("::")[0]
        parts = []
        for k, v in PREFIX_TO_PARTS.items():
            if pre.startswith(k):
                parts = v; break
        m[aid - 1] = parts
    return m


def main():
    base = ensure_extracted()
    A, paths, cls, class_names, part_locs, attr_names, parts = build(base)
    from sklearn.metrics import roc_auc_score

    # Represent each image by its species' 312-attribute concept profile
    # (CUB class-level continuous attributes, 0..100 -> [0,1]).  This is the
    # standard concept representation and denoises the per-image MTurk
    # annotations; a tiny jitter keeps same-species points distinct.
    cont = np.loadtxt(base / "attributes" / "class_attribute_labels_continuous.txt") / 100.0
    rng = np.random.default_rng(0)
    feats_all = cont[cls - 1] + rng.normal(0, 0.02, (len(cls), N_ATTR))

    idx, y, in_idx, an_idx = select_task(cls, class_names)
    X = feats_all[idx]
    # Export the AnoCUB anomaly-detection task as a standalone artifact
    # (ADBench-style X,y plus provenance) for reuse.
    task = CUB / "anocub_task.npz"
    np.savez_compressed(
        task, X=X.astype(np.float32), y=y.astype(int),
        image_paths=paths[idx],
        class_id=cls[idx],
        attr_names=np.array([attr_names[i + 1] for i in range(N_ATTR)], dtype=object))
    print(f"AnoCUB task saved -> {task}  (X={X.shape}, anomalies={int(y.sum())})")
    # drop zero-variance attributes for a cleaner scorer; keep a mask back to names
    keep = X.std(0) > 1e-9
    Xk = X[:, keep]
    expl = WANDExplainer(K=1024, seed=0, whiten=False).fit(Xk)
    s = expl.score(Xk)
    auc = roc_auc_score(y, s)
    print(f"AnoCUB: {len(in_idx)} inliers (sparrows) + {len(an_idx)} anomalies "
          f"| d={Xk.shape[1]} attrs | detection AUC = {auc:.3f}")

    # ---- rank by score: top-3 outliers (flagged) and top-3 inliers,
    #      de-duplicated to distinct species for a varied gallery ----
    a2p = attr_part_map(attr_names)
    keep_idx = np.where(keep)[0]
    order = np.argsort(s)[::-1]

    def distinct(order_seq, k=3):
        seen, out = set(), []
        for loc in order_seq:
            c = cls[idx[loc]]
            if c in seen:
                continue
            seen.add(c); out.append(loc)
            if len(out) == k:
                break
        return out

    top_out = distinct(order, 3)            # highest score = most anomalous
    top_in = distinct(order[::-1], 1)       # the single most-normal inlier

    def species(local):
        return class_names[cls[idx[local]]].split(".", 1)[-1].replace("_", " ")

    def pct(local):                          # anomaly-score percentile
        return 100.0 * float((s < s[local]).mean())

    def attribution(local):
        aw = np.zeros(N_ATTR)
        aw[keep_idx] = expl.witness_attribution(Xk[local:local + 1])[0]
        ps = {p: 0.0 for p in parts.values()}
        for j, val in enumerate(aw):
            for pn in a2p.get(j, []):
                ps[pn] += val
        return aw, ps

    print("top-3 outliers:", [(species(i), round(pct(i), 1)) for i in top_out])
    print("top-1 inlier  :", [(species(i), round(pct(i), 1)) for i in top_in])

    # ---------------- figure: top-3 outliers + top-1 inlier ------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})
    import matplotlib.image as mpimg
    import matplotlib.patheffects as pe

    bbox = {}
    for line in (base / "bounding_boxes.txt").read_text().splitlines():
        t = line.split(); bbox[int(t[0])] = [float(v) for v in t[1:5]]

    def load_crop(gid):
        im = mpimg.imread(str(base / "images" / paths[gid]))
        Hh, Ww = im.shape[:2]
        bx, by, bw, bh = bbox[gid + 1]               # image_id is 1-indexed
        pad = 0.15
        cx0 = max(0, int(bx - pad * bw)); cy0 = max(0, int(by - pad * bh))
        cx1 = min(Ww, int(bx + bw + pad * bw)); cy1 = min(Hh, int(by + bh + pad * bh))
        return im[cy0:cy1, cx0:cx1], cx0, cy0

    cmap = plt.cm.autumn_r
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.6))
    sc = None
    for j, loc in enumerate(top_out):                # flagged outliers + heatmap
        gid = idx[loc]; im, ox, oy = load_crop(gid); a = ax[j]
        a.imshow(im); a.axis("off")
        aw, ps = attribution(loc)
        smax = max(ps.values()) + 1e-9
        top_part = max(ps, key=ps.get)
        pl = part_locs[gid]
        for pid, pname in parts.items():
            x, yy, vis = pl[pid]
            if vis < 1:
                continue
            x, yy = x - ox, yy - oy
            w = ps.get(pname, 0.0) / smax
            if w <= 0.05:
                continue
            sc = a.scatter([x], [yy], s=70 + 650 * w, c=[w], cmap=cmap,
                           vmin=0, vmax=1, alpha=0.85, edgecolor="k",
                           linewidth=0.6, zorder=3)
            if pname == top_part:
                a.annotate(pname, (x, yy), fontsize=10, zorder=4, xytext=(4, 4),
                           textcoords="offset points", path_effects=halo)
        ta = attr_names[int(np.argmax(aw)) + 1].replace("has_", "").replace("::", ": ")
        ta = "\n".join(textwrap.wrap(ta, 32)) or ta
        a.set_title(f"{species(loc)}\n{pct(loc):.1f}th pct, flagged\n{ta}",
                    fontsize=9.5, color="#b30000")
    # top-1 inlier beside the outliers (normal -- nothing flagged)
    iloc = top_in[0]; gid = idx[iloc]; im, _, _ = load_crop(gid); a = ax[3]
    a.imshow(im); a.axis("off")
    a.set_title(f"{species(iloc)}\n{pct(iloc):.0f}th pct, inlier\nnormal",
                fontsize=9.5, color="#1a7d1a")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.02, wspace=0.06)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.012, pad=0.008)
        cb.set_label("part attribution", fontsize=10); cb.ax.tick_params(labelsize=9)
    out = ROOT / "figures" / "anocub_casestudy.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

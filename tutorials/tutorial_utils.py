"""Reusable helpers for the TissueNarrator tutorial notebooks.

Scope is deliberately small: only the boilerplate that repeats across notebooks
or across panels lives here. One-off figure code stays in the notebooks so the
plotting logic is visible where it is used.

Contents:
  * paths / styling
  * cell-sentence -> AnnData reconstruction (used by every downstream panel)
  * spatial proximal/control assignment and the proximal-vs-control DEG pipeline
  * the two multi-panel plots (grouped metric bars, neighborhood crop)

All data paths resolve relative to ``tutorials/data/`` (see ``data/README.md``).
"""
from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def data_path(*parts) -> str:
    """Resolve a path inside tutorials/data/."""
    return str(DATA.joinpath(*parts))


def setup_style(font_path=None):
    """Match the paper's matplotlib style (Arial if bundled, else default sans)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = Path(font_path) if font_path else (HERE / "ARIAL.TTF")
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))
        plt.rcParams["font.family"] = "arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["mathtext.fontset"] = "dejavuserif"


# -------------------------------------------------------------------
# Cell sentence -> AnnData reconstruction
# (shared by the cell-interaction LFC and the in-silico transplant panels)
# -------------------------------------------------------------------
def generate_vocabulary(adata):
    vocab = OrderedDict()
    gene_sums = np.ravel(np.sum(adata.X > 0, axis=0))
    for i, name in enumerate(adata.var_names):
        vocab[name.upper()] = gene_sums[i]
    return vocab


def gene_index(vocab_list):
    """Map gene name -> position (build once, reuse across reconstructions)."""
    return {g: i for i, g in enumerate(vocab_list)}


def post_process_genes(cell_sentence, vocab_list, top_n=100, nonsense="NOT_A_GENE"):
    """Replace non-gene words, de-duplicate genes by averaging their ranks, and
    keep the top ``top_n`` (datasets differ: MERFISH/ovarian use 100, the
    full-panel Perturb-FISH uses 200)."""
    words = cell_sentence.upper().split(" ")
    names = [w if w in vocab_list else nonsense for w in words]
    out = names.copy()
    for g, n in Counter(names).items():
        if n > 1 and g != nonsense:
            pos = [i for i, e in enumerate(out) if e == g]
            avg = int(sum(pos) / len(pos))
            out = [e for e in out if e != g]
            out.insert(avg, g)
    return " ".join(out).replace(" " + nonsense, "").split()[:top_n]


def reconstruct_expression(genes, gene_idx, slope, intercept):
    """Reconstruct an expression vector from a ranked gene list via the dataset's
    log10-rank vs log-expression linear fit. ``gene_idx`` is from gene_index()."""
    vec = np.zeros(len(gene_idx), dtype=np.float32)
    log_ranks = np.log10(1 + np.arange(len(genes)))
    for pos, g in enumerate(genes):
        j = gene_idx.get(g)
        if j is not None:
            vec[j] = intercept + slope * log_ranks[pos]
    return vec


def build_generated_adata(cell_list, obs_df, adata_ref, vocabulary, slope, intercept, top_n=100):
    """Turn generated gene-list strings into an AnnData with reconstructed
    expression. ``slope``/``intercept`` are the dataset's log-rank vs
    log-expression linear fit (define them in the notebook)."""
    import anndata
    from tqdm import tqdm
    vocab_list = list(vocabulary.keys())
    gi = gene_index(vocab_list)
    X = np.stack([
        reconstruct_expression(post_process_genes(s, vocab_list, top_n), gi, slope, intercept)
        for s in tqdm(cell_list)])
    return anndata.AnnData(X=X, obs=obs_df.copy(), var=adata_ref.var.copy())


def blocks_to_gene_strings(series):
    """Strip <cs>/</cs> tags from a column of generated blocks -> gene strings."""
    return (series.str.replace("<cs>", "", regex=False)
            .str.replace("</cs>", "", regex=False).str.strip().to_list())


# -------------------------------------------------------------------
# Proximal/control assignment + ground-truth DEG (one interaction pair)
# -------------------------------------------------------------------
def proximity_groups(adata, endo, neighbor, radius=30.0, n_hvg=500, n_control=2000):
    """Label ``endo`` cells proximal/control by distance to the nearest
    ``neighbor`` cell (< ``radius`` µm, per section), on the high-confidence test
    split. Controls are capped to ``n_control`` (seed 0). Returns
    ``(groups_df[cell_index, group], endo_adata)`` where ``endo_adata`` is the
    HVG-filtered, control-capped focal-cell AnnData carrying ``obs['is_proximal']``."""
    import scanpy as sc
    from scipy.spatial import cKDTree
    sub = adata[(adata.obs["subclass_confidence_score"] > 0.8)
                & (adata.obs["split"] == "test")
                & (adata.obs["subclass"].isin([endo, neighbor]))].copy()
    sub.obsm["spatial"] = sub.obs[["x", "y"]].to_numpy()
    m_e, m_b = (sub.obs["subclass"] == endo).to_numpy(), (sub.obs["subclass"] == neighbor).to_numpy()
    is_prox = np.zeros(sub.n_obs, dtype=bool)
    for sec in sub.obs["section"].unique():
        ms = (sub.obs["section"] == sec).to_numpy()
        me, mb = ms & m_e, ms & m_b
        if me.sum() == 0 or mb.sum() == 0:
            continue
        dmin, _ = cKDTree(sub.obsm["spatial"][mb]).query(sub.obsm["spatial"][me], k=1)
        is_prox[np.where(me)[0]] = dmin < radius
    sub.obs["is_proximal"] = is_prox
    endo_ad = sub[m_e].copy()
    sc.pp.highly_variable_genes(endo_ad, flavor="seurat_v3", n_top_genes=n_hvg, inplace=True)
    endo_ad = endo_ad[:, endo_ad.var["highly_variable"]].copy()
    prox = endo_ad.obs["is_proximal"].values
    ctrl_idx = np.where(~prox)[0]
    if len(ctrl_idx) > n_control:
        np.random.seed(0)
        keep = np.random.choice(ctrl_idx, size=n_control, replace=False)
        endo_ad = endo_ad[np.concatenate([np.where(prox)[0], keep])].copy()
    groups = pd.DataFrame({"cell_index": endo_ad.obs_names.astype(str),
                           "group": np.where(endo_ad.obs["is_proximal"], "proximal", "control")})
    return groups, endo_ad


def gt_deg(endo_ad):
    """Ground-truth proximal-vs-control DEG on the focal-cell AnnData from
    ``proximity_groups`` (genes detected in >=10 cells, Wilcoxon)."""
    import scanpy as sc
    X = endo_ad.X.toarray() if not isinstance(endo_ad.X, np.ndarray) else endo_ad.X
    ad = endo_ad[:, (X > 0).sum(axis=0) >= 10].copy()
    ad.obs["group"] = np.where(ad.obs["is_proximal"], "proximal", "non_proximal")
    sc.tl.rank_genes_groups(ad, groupby="group", groups=["proximal"],
                            reference="non_proximal", method="wilcoxon", pts=True, use_raw=False)
    return sc.get.rank_genes_groups_df(ad, group="proximal")


def predicted_tn_deg(tn_parquet, groups_df, adata_ref, vocabulary, slope, intercept):
    """TN-generated proximal-vs-control DEG for one interaction pair. Returns
    ``(deg_df, generated_adata)``."""
    import scanpy as sc
    rdf = pd.read_parquet(tn_parquet).merge(
        groups_df.rename(columns={"cell_index": "cell_name"})[["cell_name", "group"]],
        on="cell_name", how="left")
    rdf = rdf[rdf["neighbor_mode"] == "diff"]
    obs = pd.DataFrame({"group": rdf["group"].values}, index=rdf["cell_name"].values)
    g = build_generated_adata(blocks_to_gene_strings(rdf["generated_block"]),
                              obs, adata_ref, vocabulary, slope, intercept)
    X = g.X.toarray() if not isinstance(g.X, np.ndarray) else g.X
    g = g[:, (X > 0).sum(axis=0) >= 10].copy()
    sc.tl.rank_genes_groups(g, groupby="group", method="wilcoxon", pts=True, use_raw=False)
    return sc.get.rank_genes_groups_df(g, group="proximal"), g


def spearman_lfc(pred, gt, lfc_cap=10.0):
    """Spearman ρ between predicted and ground-truth LFC over shared genes."""
    from scipy.stats import spearmanr
    common = set(pred["names"]) & set(gt["names"])
    p = pred[pred["names"].isin(common)].set_index("names")
    g = gt[gt["names"].isin(common)].set_index("names")
    mask = (p["logfoldchanges"].abs() <= lfc_cap) & (g["logfoldchanges"].abs() <= lfc_cap)
    p = p[mask]
    g = g.loc[p.index]
    rho, pval = spearmanr(p["logfoldchanges"], g["logfoldchanges"])
    return p["logfoldchanges"], g["logfoldchanges"], rho, pval


# -------------------------------------------------------------------
# Multi-panel plots (each used more than once)
# -------------------------------------------------------------------
def grouped_metric_barplot(cond_df, uncond_df, methods, metric="ndcg",
                           ylabel="NDCG", ylim=(0.1, 1.1), colors=None, save=None):
    """Side-by-side conditioned/unconditioned panels with @10/@30/@50 bars per
    method. ``metric`` is 'ndcg' or 'overlap'. (Fig S2A / S2B)"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    abbr_map = {"TissueNarrator": "TN", "Class_Mean": "Class Mean", "Nearest_Neighbor": "NN"}
    abbr = [abbr_map.get(m, m) for m in methods]
    bar_width, group_spacing = 0.22, 0.35
    x = np.arange(len(methods)) * (1 + group_spacing)
    cols = [f"top_10_{metric}", f"top_30_{metric}", f"top_50_{metric}"]
    if colors is None:
        colors = (["#d0dee4", "#8facbe", "#5b8094"] if metric == "ndcg"
                  else ["#c7b3d8", "#b19dcb", "#8a79b5"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, dframe) in zip(axes, [("Conditioned Generation", cond_df),
                                          ("Unconditioned Generation", uncond_df)]):
        vals = dframe.loc[methods, cols].values
        for i in range(3):
            ax.bar(x + i * bar_width, vals[:, i], width=bar_width, color=colors[i])
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(abbr, fontsize=13, rotation=30, ha="right")
        ax.set_xlabel(title, fontsize=14)
        ax.set_ylim(*ylim)
        for pos in ("top", "right"):
            ax.spines[pos].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="y", labelsize=13, width=1.5)
    axes[0].set_ylabel(ylabel, fontsize=16)
    axes[1].legend(handles=[mpatches.Patch(color=colors[i], label=l)
                            for i, l in enumerate(["@10", "@30", "@50"])],
                   frameon=False, fontsize=13, loc="upper right")
    plt.tight_layout(pad=0.5)
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


def plot_cell_neighborhood(adata, groups_df, idx, size=100, ratio=0.65, shift=0.0,
                           subclass_a="Peri NN", color_a="#C25759",
                           subclass_b="BAM NN", color_b="#599cb4", save=None):
    """Rectangular spatial crop centered on a cell, highlighting two subclasses.
    (Fig 3B/C left, one call per interaction pair.)"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    center = groups_df.iloc[idx]["cell_index"]
    asec = adata[adata.obs["section"] == adata.obs.loc[center, "section"]].copy()
    cx, cy = asec.obs.loc[center, ["x", "y"]]
    half_h, half_w = size / 2, (size * ratio) / 2
    cxs = cx + shift * half_w
    x_min, x_max, y_min, y_max = cxs - half_w, cxs + half_w, cy - half_h, cy + half_h
    crop = asec[(asec.obs["x"] >= x_min) & (asec.obs["x"] <= x_max)
                & (asec.obs["y"] >= y_min) & (asec.obs["y"] <= y_max)].copy()
    crop.obs["c"] = crop.obs["subclass"].map({subclass_a: color_a, subclass_b: color_b}).fillna("lightgrey")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(crop.obs["x"], crop.obs["y"], c=crop.obs["c"], s=100, linewidths=0, alpha=0.9)
    ax.add_patch(Rectangle((x_min, y_min), 2 * half_w, 2 * half_h,
                           linewidth=1.2, edgecolor="black", facecolor="none"))
    a = subclass_a[:-3] if subclass_a.endswith(" NN") else subclass_a
    b = subclass_b[:-3] if subclass_b.endswith(" NN") else subclass_b
    b = "Inh IMN" if b == "OB-STR-CTX Inh IMN" else b
    off = 0.03 if a == "Astro-OLF" else 0.0
    ax.text(0.5 + off, 1.03, f"{a}  ", color=color_a, fontsize=22, ha="right", va="bottom", transform=ax.transAxes)
    ax.text(0.5 + off, 1.03, "-", color="black", fontsize=22, ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5 + off, 1.03, f"  {b}", color=color_b, fontsize=22, ha="left", va="bottom", transform=ax.transAxes)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()

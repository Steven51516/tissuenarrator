"""Spatial-QA data-construction logic for the 4 tasks reproduced in
``07_spatial_qa.ipynb``.

Ported (faithfully, trimmed to the 4 tutorial tasks) from the production
build scripts:
  - highly_expressed_genes, spatial_real_vs_fake -> data/merfish/text_qa_v2.ipynb
  - forward_spatial (de-pert), pathway_spatial (de-pathway)
        -> spatial_qa/perturb_qa/{compute_de, build_qa, templates}.py

Each ``build_*`` returns a DataFrame with at least (prompt, answer) columns,
constructed end-to-end from the underlying spatial data (no GPT / no GPU).
"""
from __future__ import annotations

import re
import string
from collections import OrderedDict

import numpy as np
import pandas as pd

CS_RE = re.compile(r"<cs>(.*?)</cs>", re.DOTALL)


# ===================================================================
# 1. highly_expressed_genes  (MERFISH)
# ===================================================================
def build_hxg_qa(adata, region_col="parcellation_structure", min_cells=20,
                 top_k=100, top_answer=20):
    """For every (region, cell class) with >= min_cells, rank genes by mean
    expression and ask for the top-20. region_col is the parcellation level."""
    obs = adata.obs
    valid = ~obs[region_col].astype(str).str.contains("unassigned", case=False, na=False)
    obs = obs[valid]
    counts = (obs.groupby([region_col, "class"], observed=True).size()
              .reset_index(name="n_cells"))
    counts = counts[counts["n_cells"] >= min_cells]

    X = adata.X
    region = obs[region_col].astype(str).to_numpy()
    klass = obs["class"].astype(str).to_numpy()
    genes = np.asarray(adata.var_names)
    keep_idx = np.where(valid.to_numpy())[0]
    Xv = X[keep_idx]

    rows = []
    for _, r in counts.iterrows():
        reg, cls = str(r[region_col]), str(r["class"])
        mask = (region == reg) & (klass == cls)
        if mask.sum() == 0:
            continue
        mean_exp = np.asarray(Xv[mask].mean(axis=0)).ravel()
        top = genes[np.argsort(mean_exp)[::-1][:top_k]]
        answer = ", ".join(list(top)[:top_answer])
        prompt = (f"List the top 20 highly expressed genes in the {cls} cell class "
                  f"of the {reg} region from the mouse brain. "
                  f"Format your answer as a ranked list of gene names separated by commas, "
                  f"for example: <ANSWER>gene1, gene2, ...")
        rows.append({"task": "highly_expressed_genes", "region": reg,
                     "cell_class": cls, "prompt": prompt, "answer": answer})
    return pd.DataFrame(rows)


# ===================================================================
# 2. spatial_real_vs_fake  (MERFISH)
# ===================================================================
def traverse_cells(coords, method="corner_nn", random_state=42):
    """Distance-weighted traversal order (same tau-style scheme as training)."""
    np.random.seed(random_state)
    n = coords.shape[0]
    if method == "random":
        return list(np.random.permutation(n))
    # corner_nn: random corner anchor, sample with prob 1/(d+eps) (near-first)
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
    anchor = corners[np.random.choice(4)]
    dists = np.linalg.norm(coords - anchor, axis=1)
    remaining, order = np.arange(n), []
    while len(remaining) > 0:
        probs = 1.0 / (dists[remaining] + 1e-6)
        probs = probs / probs.sum()
        idx = np.random.choice(remaining, p=probs)
        order.append(idx)
        remaining = remaining[remaining != idx]
    return order


def build_real_vs_fake_qa(cell_df, bin_width=200, truncate=100, min_cell_num=10,
                          max_cell_num=25, min_replace_frac=0.1, min_far_multiplier=2.0,
                          random_state=50):
    """Bin cells per section; for each bin emit a REAL spatial sentence (label 1)
    and a FAKE one (label 0) where ~min_replace_frac of the cells are swapped for
    far-away cells from the same section. Ask Yes/No 'is this real?'."""
    rng = np.random.default_rng(random_state)
    rows = []
    for section, sdf in cell_df.groupby("section"):
        sdf = sdf.copy()
        sec_coords = sdf[["x", "y"]].to_numpy(dtype=float)
        sec_names = sdf["cell_name"].to_numpy()
        sec_sent = sdf["sentence"].to_numpy()
        xb = (sdf["x"].to_numpy() // bin_width).astype(int)
        yb = (sdf["y"].to_numpy() // bin_width).astype(int)
        sdf["bin_id"] = [f"{i}_{j}" for i, j in zip(xb, yb)]
        for bin_id, g in sdf.groupby("bin_id"):
            if len(g) < min_cell_num or len(g) > max_cell_num:
                continue
            g = g.reset_index(drop=True)
            coords = g[["x", "y"]].to_numpy(dtype=float)
            order = traverse_cells(coords, "corner_nn", random_state)

            blocks, xy = [], []
            for idx in order:
                r = g.iloc[idx]
                cs = " ".join(str(r["sentence"]).split(" ")[:truncate])
                blocks.append(f"<pos> X: {int(r['x'])}, Y: {int(r['y'])} <cs> {cs} </cs>")
                xy.append((float(r["x"]), float(r["y"])))
            real_sentence = " ".join(blocks)

            # FAKE: replace a few cells with far-away cells in the same section
            n = len(blocks)
            n_rep = max(1, int(np.ceil(min_replace_frac * n)))
            fake_blocks = list(blocks)
            used = set(g["cell_name"].tolist())
            min_dist = min_far_multiplier * bin_width
            for p in rng.choice(n, size=min(n_rep, n), replace=False):
                x0, y0 = xy[p]
                d = np.sqrt((sec_coords[:, 0] - x0) ** 2 + (sec_coords[:, 1] - y0) ** 2)
                cand = np.where(d > min_dist)[0]
                if cand.size == 0:
                    continue
                pick = None
                for _ in range(50):
                    j = int(rng.choice(cand))
                    if sec_names[j] not in used:
                        pick = j
                        break
                if pick is None:
                    continue
                used.add(sec_names[pick])
                cs = " ".join(str(sec_sent[pick]).split(" ")[:truncate])
                fake_blocks[p] = (f"<pos> X: {int(sec_coords[pick,0])}, "
                                  f"Y: {int(sec_coords[pick,1])} <cs> {cs} </cs>")
            fake_sentence = " ".join(fake_blocks)

            for sent, label in ((real_sentence, 1), (fake_sentence, 0)):
                prompt = (
                    "You are given a spatial neighborhood of tissue from the mouse brain.\n"
                    "Each cell is formatted as:\n"
                    "<pos> X: <int>, Y: <int> <cs> gene1 gene2 ... </cs>\n"
                    "Here, <pos> gives the cell location and <cs> is a ranked list of gene names for that cell.\n"
                    "Decide whether this is a real spatial sentence (no cells inserted from outside the neighborhood).\n"
                    "Answer with exactly one token: Yes or No.\n\n"
                    f"Spatial sentence:\n{sent}\n")
                rows.append({"task": "spatial_real_vs_fake", "section": section,
                             "label": label, "prompt": prompt,
                             "answer": "Yes" if label == 1 else "No"})
    return pd.DataFrame(rows)


def cell_sentences_from_adata(adata, top_k=100):
    """Per-cell ranked gene list (descending expression) -> cell_df with
    section/x/y/cell_name/sentence, for the real-vs-fake builder."""
    X = adata.X
    X = X.toarray() if not isinstance(X, np.ndarray) else X
    genes = np.asarray(adata.var_names)
    order = np.argsort(-X, axis=1)[:, :top_k]
    nz = (X > 0)
    sents = []
    for i in range(X.shape[0]):
        gi = [j for j in order[i] if nz[i, j]]
        sents.append(" ".join(genes[gi]))
    return pd.DataFrame({
        "section": adata.obs["section"].astype(str).to_numpy(),
        "x": adata.obs["x"].to_numpy(), "y": adata.obs["y"].to_numpy(),
        "cell_name": adata.obs_names.astype(str).to_numpy(),
        "sentence": sents,
    })


# ===================================================================
# 3+4. Perturb-FISH DE  ->  forward_spatial (de-pert) + pathway_spatial (de-pathway)
# ===================================================================
PADJ_TH, LFC_TH, LFC_OUTLIER, MIN_GROUP = 0.05, 0.15, 15.0, 10
TOP_N_DIR = 200


def single_perturbation_labels(adata):
    """All single-gene knockout labels (no underscore, not Control). The QA DE
    is computed over ALL of these (matches the production build), not just the
    leave-one-out test perts."""
    vc = adata.obs["perturbation"].astype(str).value_counts()
    return [l for l in vc.index
            if l not in ("Control", "nan", "NA", "None", "") and "_" not in l]


def compute_perturb_de(adata, perts):
    """Per-perturbation DE: T cells near KO vs near Control (Wilcoxon).
    Returns (de_df[up/down], nonde_df). Requires obsp['spatial_connectivities']."""
    import scanpy as sc
    pert = adata.obs["perturbation"].astype(str).to_numpy()
    is_t = (adata.obs["celltype2"].astype(str) == "T cells").to_numpy()
    conn = adata.obsp["spatial_connectivities"]
    up_all, down_all, nonde_all = [], [], []
    for ko in perts:
        near_ko = np.asarray(conn @ (pert == ko).astype(np.float32)[:, None]).ravel() > 0
        near_ct = np.asarray(conn @ (pert == "Control").astype(np.float32)[:, None]).ravel() > 0
        t_ko, t_ct = is_t & near_ko, is_t & near_ct
        if t_ko.sum() < MIN_GROUP or t_ct.sum() < MIN_GROUP:
            continue
        sel = t_ko | t_ct
        sub = adata[sel].copy()
        sub.obs["group"] = np.where(t_ko[sel], "test", "control")
        sc.pp.filter_genes(sub, min_cells=5)
        sc.tl.rank_genes_groups(sub, "group", groups=["test"], reference="control",
                                method="wilcoxon", key_added="wx")
        r = sub.uns["wx"]
        df = pd.DataFrame({"gene": r["names"]["test"], "lfc": r["logfoldchanges"]["test"],
                           "padj": r["pvals_adj"]["test"]})
        df = df[df["lfc"].abs() < LFC_OUTLIER]
        up = df[(df.padj < PADJ_TH) & (df.lfc > LFC_TH)].nsmallest(TOP_N_DIR, "padj").assign(direction="up", perturbation=ko)
        down = df[(df.padj < PADJ_TH) & (df.lfc < -LFC_TH)].nsmallest(TOP_N_DIR, "padj").assign(direction="down", perturbation=ko)
        de_genes = set(up.gene) | set(down.gene)
        nonde = df[~df.gene.isin(de_genes)].assign(direction="not_de", perturbation=ko)
        up_all.append(up); down_all.append(down); nonde_all.append(nonde)
    de = pd.concat(up_all + down_all, ignore_index=True) if up_all else pd.DataFrame()
    nonde = pd.concat(nonde_all, ignore_index=True) if nonde_all else pd.DataFrame()
    return de, nonde


# ---- shared perturb-FISH prompt scaffolding (from templates.py) ----
_INTRO = ("Perturb-FISH combines MERFISH imaging with local amplification of the gRNA "
          "region to decode both genetic perturbations and the transcriptome in their "
          "spatial context. The screen below profiled a tumor xenograft in which CRISPR "
          "knockouts were applied to the cancer cells. For each knockout, we compare T "
          "cells in physical contact with KO cancer cells against T cells in contact with "
          "Control cancer cells.")
# The DE threshold quoted in the prompt is tied to the one actually used to label
# up/down (LFC_TH), so the question and the answer key never disagree.
_FWD_PREAMBLE = (_INTRO + f" A gene is called upregulated if adjusted p < {PADJ_TH} and log "
                 f"fold change > {LFC_TH}, downregulated if adjusted p < {PADJ_TH} and log fold "
                 f"change < -{LFC_TH}, and not differentially expressed otherwise.")
_PW_PREAMBLE = (_INTRO + " A pathway is called activated if its members show coordinated "
                "positive enrichment in the KO-adjacent T cells (GSEA, FDR < 0.25 with "
                "positive normalized enrichment score).")
FORWARD_LABELS = ("upregulated", "downregulated", "not differentially expressed")
DIR_TO_LABEL = {"up": "upregulated", "down": "downregulated", "not_de": "not differentially expressed"}


def _choices_block(labels, answer, rng):
    order = rng.permutation(len(labels))
    ordered = [labels[i] for i in order]
    letters = string.ascii_uppercase[:len(ordered)]
    block = "\n".join(f"{l}) {lab}" for l, lab in zip(letters, ordered))
    return block, letters[ordered.index(answer)], ordered


def _spatial_block(ko_lists, ctrl_lists, pert):
    lines = [f"Example expression profiles from {len(ko_lists)} cancer cells carrying the "
             f"{pert} knockout (genes ranked by expression, highest first):"]
    lines += [f"{i}. {g}" for i, g in enumerate(ko_lists, 1)]
    lines += ["", f"Example expression profiles from {len(ctrl_lists)} Control (non-perturbed) "
              f"cancer cells (same ranking convention):"]
    lines += [f"{i}. {g}" for i, g in enumerate(ctrl_lists, 1)]
    return "\n".join(lines) + "\n\n"


def load_cancer_pool(neighbor_df):
    """pert -> list of cancer-cell <cs> gene-list strings ('Control' = control pool)."""
    df = neighbor_df[neighbor_df["cell_type"] == "cancer"]
    pool = {}
    for pert, sub in df.groupby("perturbation"):
        gls = [m.group(1).strip() for s in sub["sentence"] if (m := CS_RE.search(str(s)))]
        pool[str(pert)] = gls
    return pool


def _sample_spatial(pool, pert, rng, n_ko=3, n_ctrl=3):
    ko, ctrl = pool.get(pert, []), pool.get("Control", [])
    if len(ko) < n_ko or len(ctrl) < n_ctrl:
        return ""
    ki = rng.choice(len(ko), n_ko, replace=False)
    ci = rng.choice(len(ctrl), n_ctrl, replace=False)
    return _spatial_block([ko[int(i)] for i in ki], [ctrl[int(i)] for i in ci], pert)


def build_forward_spatial_qa(de_df, nonde_df, pool, n_per_class=20, seed=42):
    """de-pert: classify a gene as up/down/not-DE for a KO, with 3 KO + 3 Control
    cancer-cell expression profiles appended (spatial context)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pert, sub in de_df.groupby("perturbation"):
        block = _sample_spatial(pool, pert, rng)
        if not block:
            continue
        up = sub[sub.direction == "up"].sort_values("padj").head(n_per_class)
        down = sub[sub.direction == "down"].sort_values("padj").head(n_per_class)
        nonde = nonde_df[nonde_df.perturbation == pert]
        n_nonde = min(int(round((len(up) + len(down)) / 2)), len(nonde))
        picks = [(r.gene, DIR_TO_LABEL[r.direction]) for _, r in pd.concat([up, down]).iterrows()]
        if n_nonde > 0:
            picks += [(r.gene, DIR_TO_LABEL["not_de"])
                      for _, r in nonde.sample(n=n_nonde, random_state=seed).iterrows()]
        for gene, ans in picks:
            cb, letter, ordered = _choices_block(FORWARD_LABELS, ans, rng)
            prompt = (_FWD_PREAMBLE + "\n\n" + block +
                      f"Knockout: {pert}. How does {gene} change in the neighboring T cells?\n"
                      f"{cb}\nConclude with your final answer in <answer>LETTER</answer>.")
            rows.append({"task": "forward_spatial", "perturbation": pert, "gene": gene,
                         "prompt": prompt, "answer": ans, "answer_letter": letter})
    return pd.DataFrame(rows)


def build_pathway_spatial_qa(de_df, nonde_df, pool, panel_genes, libs=None, top_n=30,
                             fdr_pos=0.25, fdr_neg=0.75, seed=42, permutation_num=100):
    """de-pathway: GSEA over the DE ranking -> activated (Yes) / unchanged (No)
    pathway calls, with spatial context. Needs network for the gene-set libraries."""
    import gseapy as gp
    rng = np.random.default_rng(seed)
    libs = libs or ["MSigDB_Hallmark_2020", "GO_Biological_Process_2025",
                    "Reactome_Pathways_2024", "KEGG_2019_Mouse"]
    full = pd.concat([de_df, nonde_df], ignore_index=True)
    big = []
    for pert, sub in full.groupby("perturbation"):
        s = sub.copy()
        s["score"] = s.lfc * -np.log10(s.padj.clip(lower=1e-6))
        ranked = s.sort_values("score", ascending=False).drop_duplicates("gene").set_index("gene")["score"]
        for lib in libs:
            try:
                res = gp.prerank(rnk=ranked, gene_sets=[lib], organism="Mouse", min_size=3,
                                 max_size=500, outdir=None, verbose=False,
                                 permutation_num=permutation_num, threads=4)
                r = res.res2d.copy()
                r["perturbation"] = pert
                r["pathway"] = r["Term"].apply(lambda x: x.split("__")[-1])
                big.append(r)
            except Exception:
                continue
    if not big:
        return pd.DataFrame()
    big = pd.concat(big, ignore_index=True)
    big["FDR"] = pd.to_numeric(big["FDR q-val"], errors="coerce")
    big["NES"] = pd.to_numeric(big["NES"], errors="coerce")
    # keep immune-relevant pathways, top-N by # perts activated
    inc = ["t cell", "immune", "cytokine", "interferon", "tnf", "nf-kb", "inflammat",
           "chemok", "interleukin", "leukocyte", "lymphocyte", "stress", "hypoxia", "apoptos"]
    big = big[big.pathway.str.lower().apply(lambda t: any(k in t for k in inc))]
    big["activated"] = (big.FDR < fdr_pos) & (big.NES > 0)
    top = big.groupby("pathway").activated.sum().sort_values(ascending=False).head(top_n).index
    big = big[big.pathway.isin(top)]
    # Keep ALL activated (Yes) calls; build a pathway-matched No pool (clean
    # unchanged, FDR>=fdr_neg, in a Yes-pathway) and sample it to len(yes),
    # guaranteeing >=1 No per perturbation that has a Yes (matches build_qa.py).
    yes = big[big.activated]
    no_pool = big[(~big.activated) & (big.FDR >= fdr_neg) & big.pathway.isin(set(yes.pathway))]
    no_idx = []
    for p in yes.perturbation.unique():
        avail = no_pool[no_pool.perturbation == p].index.tolist()
        if avail:
            no_idx.append(int(rng.choice(avail)))
    remaining = no_pool.drop(no_idx)
    need = len(yes) - len(no_idx)
    if need > 0 and len(remaining) > 0:
        no_idx += remaining.sample(n=min(need, len(remaining)), random_state=seed).index.tolist()
    balanced = pd.concat([yes, no_pool.loc[no_idx]])
    rows = []
    for _, r in balanced.iterrows():
        pert, pathway, is_yes = r.perturbation, r.pathway, bool(r.activated)
        block = _sample_spatial(pool, pert, rng)
        if not block:
            continue
        pg = panel_genes.get(pathway, "(see panel)")
        prompt = (_PW_PREAMBLE + "\n\n" + block + f"Pathway: {pathway}\n"
                  f"Pathway members in our 154-gene panel: {pg}\n"
                  f"Question: In T cells in physical contact with {pert}-knockout cancer cells "
                  f"(compared to T cells near Control cancer cells), is the {pathway} pathway activated?\n"
                  f"A) Yes\nB) No\nProvide your final answer in <answer>LETTER</answer>.")
        rows.append({"task": "pathway_spatial", "perturbation": pert, "pathway": pathway,
                     "prompt": prompt, "answer": "Yes" if is_yes else "No",
                     "answer_letter": "A" if is_yes else "B"})
    return pd.DataFrame(rows)

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, Counter
from sklearn.utils import shuffle
from scipy import sparse

def generate_vocabulary(adata):
    """
    Create a vocabulary dictionary, where each key represents a single gene
    token and the value represents the number of non-zero cells in the provided
    count matrix.

    Arguments:
        adata: an AnnData object to generate cell sentences from. Expects that
            `obs` correspond to cells and `vars` correspond to genes
    Return:
        a dictionary of gene vocabulary
    """
    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({})... "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    vocabulary = OrderedDict()
    gene_sums = np.ravel(np.sum(adata.X > 0, axis=0))

    for i, name in enumerate(adata.var_names):
        vocabulary[name.upper()] = gene_sums[i]  # keys are all uppercase gene names

    return vocabulary


def concat_vocabularies(vocabulary_list):
    """
    Helper function to concatenate multiple vocabulary ordered dictionaries.
    Preserves order of features in the first vocabulary, and appends any additional
    features from successive dictionaries.
    """
    concat_vocab = OrderedDict()
    for ordered_dict in vocabulary_list:
        for key, val in ordered_dict.items():
            if key not in concat_vocab:
                concat_vocab[key] = val
            else:
                concat_vocab[key] = concat_vocab[key] + val
    return concat_vocab


def generate_cell_sentences(adata, vocab, delimiter=' ', random_state=42):
    """
    Transform expression matrix to sentences. Sentences contain gene "words"
    denoting genes with non-zero expression. Genes are ordered from highest
    expression to lowest expression.

    Arguments:
        adata: an AnnData object to generate cell sentences from. Expects that
            `obs` correspond to cells and `vars` correspond to genes
        vocab: an OrderedDict which as feature names as keys and counts as values
        random_state: sets the numpy random state for splitting ties
    
    Returns:
        a `numpy.ndarray` of sentences, split by delimiter.
    """
    np.random.seed(random_state)

    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({}), "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    mat = sparse.csr_matrix(adata.X)
    enc_map = list(vocab.keys())

    sentences = []
    for i in tqdm(range(mat.shape[0])):
        cols = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        vals = mat.data[mat.indptr[i] : mat.indptr[i + 1]]
        cols, vals = shuffle(cols, vals)
        sentences.append(delimiter.join([enc_map[x] for x in cols[np.argsort(-vals, kind="stable")]]))

    return sentences


def build_cell_df(cell_names, sentences, adata, label_col_names=None):
    """
    Build a dataframe dataset from cell names, sentences, and adata metadata.

    Args:
        cell_names (List[str]): List of cell names (must match adata.obs_names).
        sentences (List[str]): Corresponding list of cell sentences.
        adata (AnnData): AnnData object containing metadata in `.obs`.
        label_col_names (List[str] or None): Optional list of column names in `adata.obs` to include as labels.

    Returns:
        pd.DataFrame: A dataframe with columns ["cell_name", "sentence", *label_col_names].
    """
    if len(cell_names) != len(sentences):
        raise ValueError("cell_names and sentences must be the same length.")
    
    df = pd.DataFrame({
        "cell_name": cell_names,
        "sentence": sentences
    })

    if label_col_names:
        label_df = adata.obs.loc[cell_names, label_col_names].reset_index(drop=True)
        df = pd.concat([df, label_df], axis=1)

    return df
    

def traverse_cells(coords: np.ndarray,
                   method="nn",
                   random_state=42) -> list[int]:
    
    np.random.seed(random_state)
    n = coords.shape[0]
    coords = np.asarray(coords)

    if method == "random":
        order = list(np.random.permutation(n))

    elif method in {"outward_inward", "inward_outward"}:
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        weights = dists if method == "outward_inward" else -dists
        remaining = np.arange(n)
        order = []
        while len(remaining) > 0:
            w = weights[remaining]
            probs = np.exp(w - w.max())
            probs = probs / probs.sum()
            idx = np.random.choice(remaining, p=probs)
            order.append(idx)
            remaining = remaining[remaining != idx]

    elif method == "nn":
        tau = 1.0
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        center = np.array([[coords[:, 0].mean(), coords[:, 1].mean()]])
        corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
        anchors = np.vstack([corners, center])  # 4 corners + 1 center
        anchor = anchors[np.random.choice(len(anchors))]
        dists = np.linalg.norm(coords - anchor, axis=1)
        remaining = np.arange(n)
        order = []
    
        while len(remaining) > 0:
            d = dists[remaining].astype(float)
            weights = 1.0 / ((d + 1e-6) ** tau)  # tau <1 = more random, tau>1 = greedier
            probs = weights / weights.sum()
            idx = np.random.choice(remaining, p=probs)
            order.append(idx)
            remaining = remaining[remaining != idx]

    else:
        raise ValueError(f"Unknown method: {method}")

    return order



def build_spatial_df(
    cell_df,
    bin_width=200,
    traversal_methods=("corner_nn"),
    truncate=100,
    delimiter=" ",
    random_state=42,
    meta=None,              # None or list[str]
    n_repeats=1,            # number of augmented traversals per bin
    min_cell_num=5,
    sentence_meta=None,
):
    """
    Build spatial-level sentences by grouping cells into spatial bins within each section
    and serializing cells using one or more traversal methods (data augmentation).

    Args:
        cell_df: DataFrame with columns ["cell_name", "sentence", "x", "y", "section"].
                 - "sentence" is the per-cell CS string.
        bin_width: width of square bins in spatial coordinates.
        traversal_methods: iterable of traversal strategy names.
        truncate: max number of tokens taken from each cell sentence.
        delimiter: token delimiter when joining the full spatial sentence.
        random_state: seed for reproducibility.
        meta: None or list[str]; if list, include these columns as "key: val" in <meta>.
              Also returned aligned with cell_names.
        n_repeats: number of augmented traversals per bin (>=1).

    Returns:
        DataFrame with columns:
          ["spatial_bin_id", "section", "sentence", "cell_names", "cell_meta",
           "n_cells", "traversal", "repeat_idx"]
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    rng = np.random.default_rng(random_state)
    rows = []

    required_cols = {"cell_name", "sentence", "x", "y", "section"}
    missing = required_cols - set(cell_df.columns)
    if missing:
        raise ValueError(f"cell_df is missing required columns: {sorted(missing)}")

    if meta is not None:
        if isinstance(meta, (list, tuple)):
            meta_cols = list(meta)
        else:
            raise ValueError("meta must be None or a list/tuple of column names")
        meta_missing = set(meta_cols) - set(cell_df.columns)
        if meta_missing:
            raise ValueError(f"meta columns not found in cell_df: {sorted(meta_missing)}")
    else:
        meta_cols = None

    # binning per section
    for section, section_df in tqdm(cell_df.groupby("section"), desc="Processing sections"):
        x_bin = (section_df["x"].to_numpy() // bin_width).astype(int)
        y_bin = (section_df["y"].to_numpy() // bin_width).astype(int)
        bin_ids = [f"{i}_{j}" for i, j in zip(x_bin, y_bin)]

        section_df = section_df.copy()
        section_df["bin_id"] = bin_ids

        for bin_id, group in section_df.groupby("bin_id"):
            if len(group) < min_cell_num:
                continue

            coords = group[["x", "y"]].to_numpy()

            # choose traversal methods for augmentation
            if n_repeats <= len(traversal_methods):
                picked = rng.choice(traversal_methods, size=n_repeats, replace=False)
            else:
                picked = rng.choice(traversal_methods, size=n_repeats, replace=True)

            for r_idx, method in enumerate(picked):
                order = traverse_cells(coords, method=method, random_state=random_state + r_idx)
                parts = []
                ordered_cell_names = []
                ordered_meta = []

                # ---- build sentence-level meta prefix ----
                if sentence_meta is not None:
                    meta_items = []
                    for k in sentence_meta:
                        vals = group[k].unique()
                        if len(vals) != 1:
                            raise ValueError(
                                f"Sentence-level meta column '{k}' is not unique within bin {section}__{bin_id}: {vals}"
                            )
                        v = vals[0]
                        v_str = "N/A" if pd.isna(v) else str(v)
                        meta_items.append(f"{k}: {v_str}")
                    sentence_meta_str = ", ".join(meta_items) + " "
                else:
                    sentence_meta_str = ""

                for idx in order:
                    r = group.iloc[idx]
                    pos_str = f"<pos> X: {int(round(r['x']))}, Y: {int(round(r['y']))}"
                    if meta_cols:
                        meta_items = []
                        meta_tuple = []
                        for k in meta_cols:
                            v = r[k]
                            v_str = "N/A" if pd.isna(v) else str(v)
                            meta_items.append(f"{k}: {v_str}")
                            meta_tuple.append(v_str)
                        meta_str = " <meta> " + ", ".join(meta_items)
                        ordered_meta.append(tuple(meta_tuple))
                    else:
                        meta_str = ""
                        ordered_meta.append(None)

                    cs_tokens = str(r["sentence"]).split(delimiter)[:truncate]
                    cs_str = " ".join(cs_tokens)
                    block = f"{pos_str}{meta_str} <cs> {cs_str} </cs>"
                    parts.append(block)
                    ordered_cell_names.append(r["cell_name"])

                spatial_sentence = sentence_meta_str + delimiter.join(parts)
                rows.append({
                    "spatial_bin_id": f"{section}__{bin_id}",
                    "section": section,
                    "sentence": spatial_sentence,
                    "cell_names": ordered_cell_names,
                    "cell_meta": ordered_meta, 
                    "n_cells": len(ordered_cell_names),
                    "traversal": method,
                    "repeat_idx": r_idx, 
                })

    return pd.DataFrame(rows)
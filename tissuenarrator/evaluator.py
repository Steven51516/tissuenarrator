import re, math, random, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Iterable, Tuple, Optional, Dict
import math
import collections

from .data import *

def dcg(relevances: Iterable[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg(pred_ranked: List[str], truth_ranked: List[str]) -> float:
    truth = set(truth_ranked)
    rel   = [1.0 if g in truth else 0.0 for g in pred_ranked]
    ideal = sorted(rel, reverse=True)
    num, den = dcg(rel), dcg(ideal)
    return num / den if den > 0 else 0.0

def remove_meta_between_tags(df: pd.DataFrame, column: str) -> pd.DataFrame:
    pattern = r'<meta>.*?<cs>'
    df = df.copy()
    df[column] = df[column].apply(lambda x: re.sub(pattern, '<cs>', x))
    return df
    
class SpatialEvaluator:
    def __init__(self, model):
        self.model  = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _cells(self, text: str) -> List[CellSentence]:
        if not isinstance(text, str) or not text.strip():
            return []
        try:
            return SpatialSentence.from_string(text, strict=True).cells
        except Exception:
            return []

    def _first_cell(self, text: str) -> Optional[CellSentence]:
        cs = self._cells(text)
        return cs[0] if cs else None

    def _header_until_cs(self, cell: Optional[CellSentence]) -> str:
        if not cell:
            return ""
        return cell.to_string(include=("pos", "meta")).strip() + " <cs>"

    def _genes_from_block(self, block: str) -> List[str]:
        c = self._first_cell(block)
        return list(c.cs) if c else []

    def _cut_after_cs(self, s: str) -> str:
        m = re.search(r"</\s*cs\s*>", s, flags=re.IGNORECASE)
        return s if not m else s[: m.end()]

    def _reorder_prompt_by_distance(
        self,
        prompt_text: str,
        target_cell: Optional[CellSentence],
        order: Optional[str] = "prob_far_to_near",
        total_prompt_cells: Optional[int] = None,
        tau: float = 1.0,
    ) -> str:
        sp = SpatialSentence.from_string(prompt_text, strict=True)
        if not sp.cells or target_cell is None:
            return prompt_text

        if order is not None:
            tx, ty = target_cell.xy()
            def dist(c: CellSentence) -> float:
                cx, cy = c.xy()
                return math.dist((tx, ty), (cx, cy))

            if order == "prob_far_to_near":
                # tau-regulated random far->near traversal (matches the training-time
                # nn traversal): sample cells without replacement with probability
                # proportional to dist**tau, so far cells tend to come first and the
                # closest cells end up last (next to the target). tau controls how
                # strongly distance is favoured.
                remaining, ordered = sp.cells[:], []
                while remaining:
                    weights = [(dist(c) + 1e-8) ** tau for c in remaining]
                    chosen = random.choices(remaining, weights=weights, k=1)[0]
                    ordered.append(chosen)
                    remaining.remove(chosen)
                sp.cells = ordered
            elif order == "random":
                random.shuffle(sp.cells)
            else:  # deterministic "far_to_near" / "near_to_far"
                sp.cells = sorted(sp.cells, key=dist, reverse=(order == "far_to_near"))

        # apply trim even if no sorting
        if total_prompt_cells is not None and total_prompt_cells > 0:
            sp.cells = sp.cells[-total_prompt_cells:]

        return sp.to_string(include_sentence_meta=True, include_cell=("pos", "meta", "cs"))

    def inference_batch(
        self,
        prompts: List[str],
        answers: List[str],
        max_new_tokens: int = 400,
        mode: str = "meta_all",
        reorder_by_xy: Optional[str] = "prob_far_to_near",
        total_prompt_cells: Optional[int] = None,
        tau: float = 1.0,
        **kwargs,
    ) -> List[str]:
        seeds, headers = [], []
        for prompt, answer in zip(prompts, answers):
            ans_cell = self._first_cell(answer)
            prompt_sorted = self._reorder_prompt_by_distance(prompt, ans_cell, order=reorder_by_xy, total_prompt_cells=total_prompt_cells, tau=tau)
            header = self._header_until_cs(ans_cell)
            if mode == "meta_neighbor" and ans_cell:
                header = f"<pos> X: {ans_cell.x}, Y: {ans_cell.y} <meta>"
            seed = (prompt_sorted.rstrip() + " " + header).strip()
            seeds.append(seed)
            headers.append(header)

        outs = self.model.inference_batch(seeds, max_new_tokens=max_new_tokens, **kwargs)
        return [(h + " " + self._cut_after_cs(o)).strip() for h, o in zip(headers, outs)]
        
    def _evaluate_block(self, generated_block: str, answer: str, top_k_list=(10, 30, 50)) -> Dict[str, float]:
        pred = self._genes_from_block(generated_block)
        true = self._genes_from_block(answer)
    
        out = dict(
            n_pred=len(pred),
            n_true=len(true),
            pred_genes=pred[:100],
            true_genes=true[:100],
        )
    
        # Precompute NDCG over the full list once
        out["ndcg"] = ndcg(pred, true)
    
        # Compute top-k metrics
        for k in top_k_list:
            tp_set, tt_set = set(pred[:k]), set(true[:k])
            inter_k = tp_set & tt_set
            out[f"top_{k}_overlap"] = len(inter_k) / max(len(tp_set), 1)
            out[f"top_{k}_ndcg"] = ndcg(pred[:k], true[:k])
    
        return out


    def evaluate_prompt_df(
        self,
        prompt_df: pd.DataFrame,
        mode: str = "meta_all",
        max_rows: int = 100,
        top_k_list=(10, 30, 50),
        greedy: bool = False,
        batch_size: int = 32,
        max_new_tokens: int = 400,
        reorder_by_xy: Optional[str] = "prob_far_to_near",
        total_prompt_cells: Optional[int] = 100,
        tau: float = 1.0,
        **kwargs,
    ):

        valid_modes = {"meta_all", "meta_neighbor", "pos_only"}
        assert mode in valid_modes, f"Invalid mode '{mode}'. Must be one of {valid_modes}."

        if mode == "no_meta":
            prompt_df = remove_meta_between_tags(prompt_df.copy(), "prompt")
            prompt_df = remove_meta_between_tags(prompt_df, "answer")

        rows, gens = [], []
        it = prompt_df.itertuples(index=True)
        batch_prompts, batch_answers = [], []
        total = min(max_rows, len(prompt_df))

        for i, r in tqdm(list(zip(range(total), it)), total=total):
            batch_prompts.append(r.prompt)
            batch_answers.append(r.answer)
            flush = (len(batch_prompts) == batch_size) or (i == total - 1)
            if flush:
                gblocks = self.inference_batch(
                    batch_prompts, batch_answers,
                    max_new_tokens=max_new_tokens,
                    mode=mode, greedy=greedy,
                    reorder_by_xy=reorder_by_xy,
                    total_prompt_cells=total_prompt_cells,
                    tau=tau,
                    **kwargs
                )
                for j, gb in enumerate(gblocks):
                    metrics = self._evaluate_block(gb, batch_answers[j], top_k_list=top_k_list)
                    metrics["row_idx"] = r.Index
                    rows.append(metrics)
                    gens.append(gb)
                batch_prompts, batch_answers = [], []

        mdf = pd.DataFrame(rows)
        overall = dict(
            ndcg      = mdf["ndcg"].mean()  if not mdf.empty else 0.0,
        )
        for k in top_k_list:
            for suf in [f"top_{k}_overlap", f"top_{k}_ndcg"]:
                overall[suf] = mdf[suf].mean() if not mdf.empty else 0.0
        return mdf, overall, gens

    # ------------------------------------------------------------------
    # Simple, model-free reference baselines (overlap@k + ndcg@k only).
    # These do NOT call the LLM; they predict a gene ranking for the target
    # cell from the prompt neighbourhood / class statistics, then score it
    # with the exact same _evaluate_block used for TissueNarrator.
    # ------------------------------------------------------------------
    def nearest_neighbor_block(
        self,
        prompt: str,
        answer: str,
        neighbor_filter: Optional[str] = None,
        majority_radius: float = 50.0,
    ) -> str:
        """Nearest-Neighbor baseline: predict the gene ranking of a single prompt
        cell chosen by spatial proximity to the target.

        neighbor_filter:
            None       -> closest prompt cell overall (no peek at the target's
                          class; used for the unconditioned setting).
            "majority" -> within `majority_radius` of the target, take the locally
                          dominant class, then the closest cell of that class
                          (used for the conditioned setting). Falls back to the
                          single closest cell if no neighbour is in radius.
            "class"    -> closest prompt cell that shares the target cell's own
                          class (uses the target's metadata directly).

        Returns a generated block string compatible with `_evaluate_block`.
        """
        p_cells = self._cells(prompt)
        a_cell = self._first_cell(answer)
        if not p_cells or a_cell is None:
            return ""
        txy = a_cell.xy()
        cand = p_cells
        if neighbor_filter == "majority":
            dists = [(math.dist(txy, c.xy()), c) for c in p_cells]
            in_radius = [(d, c) for d, c in dists if d <= majority_radius]
            if in_radius:
                classes = [c.class_lower() for _, c in in_radius]
                maj = collections.Counter([c for c in classes if c]).most_common(1)
                maj = maj[0][0] if maj else None
                same = [c for _, c in in_radius if c.class_lower() == maj]
                cand = same if same else [c for _, c in in_radius]
            else:
                cand = [min(dists, key=lambda x: x[0])[1]]
        elif neighbor_filter == "class":
            a_cls = a_cell.class_lower()
            if a_cls:
                same = [c for c in p_cells if c.class_lower() == a_cls]
                if same:
                    cand = same
        chosen = min(cand, key=lambda c: math.dist(txy, c.xy()))
        return (self._header_until_cs(a_cell) + " " + " ".join(chosen.cs) + " </cs>").strip()

    def _dominant_neighbor_class(
        self,
        prompt: Optional[str],
        neighbor_cell_names=None,
        cell2class: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        # Prefer classes embedded in the prompt metadata (conditioned prompts);
        # fall back to an external cell -> class map keyed by neighbor cell names
        # (needed for unconditioned/pos_only prompts that carry no metadata).
        classes = [c.class_lower() for c in self._cells(prompt)] if prompt else []
        classes = [c for c in classes if c]
        if not classes and neighbor_cell_names is not None and cell2class is not None:
            classes = [str(cell2class[str(n)]).strip().lower()
                       for n in list(neighbor_cell_names) if str(n) in cell2class]
        return collections.Counter(classes).most_common(1)[0][0] if classes else None

    def class_mean_block(
        self,
        answer: str,
        class_mean_table: Dict[str, str],
        *,
        prompt: Optional[str] = None,
        neighbor_cell_names=None,
        cell2class: Optional[Dict[str, str]] = None,
        class_source: str = "target",
    ) -> str:
        """Class-Mean baseline: predict a cell class's mean-expression-ranked
        gene list (built with `build_class_mean_table`).

        class_source:
            "target"   -> the target cell's own class, read from the answer
                          header (conditioned setting).
            "neighbor" -> the dominant class among the target's spatial
                          neighbours (unconditioned setting; no peek at the
                          target's own class). Neighbour classes come from the
                          prompt metadata when present, else from
                          `neighbor_cell_names` + `cell2class`.
        """
        a_cell = self._first_cell(answer)
        if a_cell is None:
            return ""
        if class_source == "target":
            cls = a_cell.class_lower()
        elif class_source == "neighbor":
            cls = self._dominant_neighbor_class(prompt, neighbor_cell_names, cell2class)
        else:
            raise ValueError("class_source must be 'target' or 'neighbor'")
        genes = class_mean_table.get(cls, "") if cls else ""
        return (self._header_until_cs(a_cell) + " " + str(genes).upper() + " </cs>").strip()

    def evaluate_baseline_df(
        self,
        prompt_df: pd.DataFrame,
        baseline: str = "nn",
        max_rows: Optional[int] = None,
        top_k_list=(10, 30, 50),
        neighbor_filter: Optional[str] = None,
        class_mean_table: Optional[Dict[str, str]] = None,
        class_source: str = "target",
        cell2class: Optional[Dict[str, str]] = None,
    ):
        """Score a model-free baseline over a prompt dataframe, returning
        (metrics_df, overall, generated_blocks) exactly like `evaluate_prompt_df`
        (overlap@k + ndcg@k only).

        baseline:
            "nn"         -> nearest_neighbor_block (see `neighbor_filter`).
            "class_mean" -> class_mean_block (needs `class_mean_table`; see
                            `class_source`, and `cell2class` for the neighbor
                            variant on metadata-free prompts).
        """
        n = len(prompt_df) if max_rows is None else min(max_rows, len(prompt_df))
        has_nb = "neighbor_cell_names" in prompt_df.columns
        rows, gens = [], []
        for i, r in tqdm(list(zip(range(n), prompt_df.itertuples(index=True))), total=n):
            if baseline == "nn":
                blk = self.nearest_neighbor_block(r.prompt, r.answer, neighbor_filter=neighbor_filter)
            elif baseline == "class_mean":
                if class_mean_table is None:
                    raise ValueError("class_mean baseline needs class_mean_table=...")
                blk = self.class_mean_block(
                    r.answer, class_mean_table, prompt=r.prompt,
                    neighbor_cell_names=(getattr(r, "neighbor_cell_names") if has_nb else None),
                    cell2class=cell2class, class_source=class_source,
                )
            else:
                raise ValueError("baseline must be 'nn' or 'class_mean'")
            m = self._evaluate_block(blk, r.answer, top_k_list=top_k_list)
            m["row_idx"] = r.Index
            rows.append(m)
            gens.append(blk)

        mdf = pd.DataFrame(rows)
        overall = dict(ndcg=mdf["ndcg"].mean() if not mdf.empty else 0.0)
        for k in top_k_list:
            for suf in [f"top_{k}_overlap", f"top_{k}_ndcg"]:
                overall[suf] = mdf[suf].mean() if not mdf.empty else 0.0
        return mdf, overall, gens


def build_class_mean_table(
    adata,
    class_key: str = "class",
    split_key: Optional[str] = "split",
    split: Optional[str] = "train",
    top_n: int = 100,
) -> Dict[str, str]:
    """Rank genes by mean expression within each cell class and return
    {class_lower -> space-joined top-N gene symbols}. By default the ranking is
    computed on the training split (`adata.obs[split_key] == split`); pass
    split_key=None to use all cells. Used by the Class-Mean baseline.
    """
    sub = adata
    if split_key is not None and split is not None and split_key in adata.obs:
        sub = adata[adata.obs[split_key] == split]
    genes = np.asarray(sub.var_names)
    classes = sub.obs[class_key].astype(str).values
    X = sub.X
    table: Dict[str, str] = {}
    for c in pd.unique(classes):
        m = np.asarray(X[classes == c].mean(axis=0)).ravel()
        table[c.strip().lower()] = " ".join(genes[np.argsort(-m)[:top_n]])
    return table

 

  

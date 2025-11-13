import re, math, torch
import pandas as pd
from tqdm import tqdm
from typing import List, Iterable, Tuple, Optional, Dict
import math
import collections

from data import *

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
        order: Optional[str] = "far_to_near",
        total_prompt_cells: Optional[int] = None,
    ) -> str:
        sp = SpatialSentence.from_string(prompt_text, strict=True)
        if not sp.cells or target_cell is None:
            return prompt_text
    
        if order is not None:
            tx, ty = target_cell.xy()
            def dist(c: CellSentence) -> float:
                cx, cy = c.xy()
                return math.dist((tx, ty), (cx, cy))
            reverse = (order == "far_to_near")
            sp.cells = sorted(sp.cells, key=dist, reverse=reverse)
    
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
        reorder_by_xy: Optional[str] = "far_to_near",
        total_prompt_cells: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        seeds, headers = [], []
        for prompt, answer in zip(prompts, answers):
            ans_cell = self._first_cell(answer)
            prompt_sorted = self._reorder_prompt_by_distance(prompt, ans_cell, order=reorder_by_xy, total_prompt_cells=total_prompt_cells)
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
        reorder_by_xy: Optional[str] = "far_to_near",
        total_prompt_cells: Optional[int] = 100,
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

 

  

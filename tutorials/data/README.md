# Tutorial data

The notebooks read every file from this folder using **relative paths only**
(`tutorials/data/...`). The large files are **not** stored in git — download them
from the shared Google Drive and drop them here.

> **Download:** <ADD GOOGLE DRIVE LINK HERE>

## MERFISH (mouse-brain ABCA Zhuang) — `01_preprocess`, `02_train`, `03_demo_inference`, `04_merfish_eval`

| File (put here) | What it is | Source on the lab server |
| --- | --- | --- |
| `merfish_preprocessed.h5ad` | Preprocessed AnnData: 2.6 M cells × 1122 genes, `obs` has `x,y,section,class,subclass,spatial_domain,split,subclass_confidence_score`. Used for class-mean stats, in-situ maps, and ground-truth LFC. | `data/merfish/merfish_preprocessed.h5ad` |
| `merfish_spatial_sentences.parquet` | Spatial-sentence training table (one row per spatial bin). Used by `01_preprocess` / `02_train`. | `data/merfish/merfish_all_spatial_df.parquet` |
| `merfish_test_cond.parquet` | **Conditioned** eval set (held-out OOD region, `meta_all`): per-cell `prompt`, `answer`, `neighbor_cell_names`, plus TissueNarrator's saved `generated_block`. | `…/results/central_ood_meta_all_final_rows0-1196.parquet` (cleaned) |
| `merfish_test_uncond.parquet` | **Unconditioned** eval set (in-distribution central; `meta_neighbor` — neighbours' metadata visible, the target's own hidden): same columns. | `…/results/merfish_central_meta_neighbor_final_rows0-15271.parquet` (cleaned) |
| `merfish_class_mean_genes.parquet` | Precomputed Class-Mean lookup: per class, the top-100 mean-expression genes (train split). Convenience — the notebook can also rebuild it from the `.h5ad`. | derived |
| `merfish_cell2class.parquet` | `cell_name → class` map (all cells). Lets the unconditioned Class-Mean baseline resolve the dominant neighbor class without loading the 12 GB `.h5ad`. | derived from `.h5ad` `obs` |
| `merfish_demo_sample.parquet` | 100-cell sample for the quick `03_demo_inference.ipynb` (kept in git). | `src_github/tutorials/merfish_demo_sample100.parquet` |

### `cell_interaction/` (cell-interaction + transplant)

| File | What it is |
| --- | --- |
| `Peri_NN_vs_BAM_NN_tn.parquet`, `Astro-OLF_NN_vs_OB-STR-CTX_Inh_IMN_tn.parquet` | TissueNarrator's generated cells for each pair (`cell_name`, `generated_block`, `answer`, `neighbor_mode`). |
| `transplant_OEC_THGlut_tn.parquet` | TissueNarrator's generated cells for the in-silico OEC↔TH-Glut transplant. |

The proximal/control assignment is **computed in the notebook** (`tu.proximity_groups`) from the `.h5ad`, so no proximity CSVs are needed.

## Ovarian (Visium-HD) — `05_ovarian_eval`

| File (put here) | What it is | Source on the lab server |
| --- | --- | --- |
| `ovarian_preprocessed.h5ad` | Preprocessed AnnData (276 k cells × 979 genes, log1p10): `obs` has `x,y,section,cell_type`. Used for marker AUC, GSEA, and the spatial map. | `data/ovarian/adata_preprocessed_log1p10.h5ad` |
| `ovarian_spatial_sentences.parquet` | Spatial-sentence training table (radius-30 neighborhoods). For `01_preprocess` / `02_train`. | `data/ovarian/ovarian_spatial_sentences_radius30.parquet` |
| `ovarian_generation.parquet` | TissueNarrator's generated cells across immune cell types (`center_cell`, `center_type`, `section`, `n_neighbors`, `n_malignant`, `generated_block`, `answer`); the 6 per-cell-type result files concatenated + cleaned. The high/low-malignancy `above_median` split is computed in the notebook from `n_malignant`. | `…/ovarian_results_2000/temp0.3_df_with_gen_and_metrics_*_nometa.parquet` |

## Perturb-FISH (THP1 tumor screen) — `06_pertfish_eval`

| File (put here) | What it is | Source on the lab server |
| --- | --- | --- |
| `pertfish_tumors_test.h5ad` | Ground-truth AnnData (spatial coords in px, `obs` has `perturbation`, `celltype2`). Used for the spatial map and ground-truth LFC. | `/work/magroup/shared/spatial_perturbation/data/perturb_fish/tumors_qc_test.h5ad` |
| `pertfish_lfc_raw.parquet` | TissueNarrator's generated neighbor-T-cell sentences per knockout (`tid`, `is_control`, `k5` = 5 samples, `pert`); the 11 per-KO `*_nz_both_raw` files concatenated. | `…/pertfish_main/fullpanel_eval/{KO}_nz_both_raw.parquet` |

## Spatial QA — `07_spatial_qa`

| File (put here) | What it is | Source on the lab server |
| --- | --- | --- |
| `qa_results.csv` | Per-task QA scores for every model (overlap@k / accuracy). The tutorial plots `v2_tn` (TN-sft) vs `v2_base` (Base-sft); the external baseline columns are also present. | `…/spatial_qa/perturb_qa/baselines/final_table_v2.csv` |
| `pertfish_neighbor.parquet` | Cancer-cell + T-cell spatial sentences per perturbation; supplies the cancer-cell expression profiles used as spatial context when constructing the forward/pathway QA items. | `data/perturb-fish/perturb_neighbor.parquet` |

The QA *data construction* in `07_spatial_qa.ipynb` also reuses `merfish_preprocessed.h5ad`
(highly_expressed_genes, spatial_real_vs_fake) and `pertfish_tumors_test.h5ad`
(forward_spatial, pathway_spatial), both already listed above.

## Model checkpoint (optional — only needed to re-run TissueNarrator inference)

The eval notebooks reconstruct metrics from the saved `generated_block` column, so
**no GPU/model is required to reproduce the figures**. To generate predictions
yourself, download the merged checkpoint and load it with `tissuenarrator.llm`:

| File | What it is |
| --- | --- |
| `merfish_epoch3_merged/` | Merged vLLM-loadable TissueNarrator (Qwen3-4B-Base + MERFISH LoRA, epoch 3). |

---
*Naming note:* `test_cond` = conditioned (model sees the target cell's metadata),
`test_uncond` = unconditioned (the target's own metadata is hidden; neighbours'
metadata is still visible). These replace the old
`central_ood` / `central_test` names.

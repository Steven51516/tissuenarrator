# Baselines

The tutorials use only the built-in Nearest-Neighbor and Class-Mean baselines. The
external baselines from the paper are listed here for reference.

## Spatial gene-expression prediction

Mouse→human symbol mapping, then an MLP head is trained after the pooling layer to
predict gene expression.

| Method | Repo |
| --- | --- |
| scGPT | https://github.com/bowang-lab/scGPT |
| Geneformer | https://huggingface.co/ctheodoris/Geneformer |
| Nicheformer | https://github.com/theislab/nicheformer |

## In-silico perturbation (Perturb-FISH)

| Method | Repo | Note |
| --- | --- | --- |
| Celcomen | https://github.com/Teichlab/celcomen | |
| CONCERT | https://github.com/mims-harvard/CONCERT | |
| MintFlow | https://github.com/Lotfollahi-lab/mintflow | The perturbed cell is encoded as a new cell type. |

## Spatial QA

| Method | Repo |
| --- | --- |
| GPT-5 | OpenAI |
| Qwen | https://huggingface.co/Qwen/Qwen3-4B-Base |
| Biomni | https://github.com/snap-stanford/Biomni |
| STAgent | https://github.com/LiuLab-Bioelectronics-Harvard/STAgent |
| SpatialAgent | https://github.com/Genentech/SpatialAgent |

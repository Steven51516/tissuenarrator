"""TissueNarrator: generative modeling of spatial transcriptomics with LLMs.

Public API:
    data       -- CellSentence / SpatialSentence spatial-sentence parsing
    preprocess -- build spatial-sentence dataframes from AnnData
    train      -- LoRA fine-tuning entry point
    llm        -- vLLM inference wrapper
    evaluator  -- SpatialEvaluator + model-free baselines (NN, Class-Mean)
"""

from .data import CellSentence, SpatialSentence
from .evaluator import (
    SpatialEvaluator,
    build_class_mean_table,
    ndcg,
)

__all__ = [
    "CellSentence",
    "SpatialSentence",
    "SpatialEvaluator",
    "build_class_mean_table",
    "ndcg",
]

__version__ = "0.1.0"

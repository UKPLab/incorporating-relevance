import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union

from sentence_transformers import SentenceTransformer
from settings import dataset_settings_cls


class ScoringFunction(str, Enum):
    cos = "cos"
    dot = "dot"


class FTParams(str, Enum):
    full = "full"
    bias = "bias"
    # adapter = "adapter"
    head = "head"


@dataclass(kw_only=True)
class Experiment:
    prefix: str
    dataset: str
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    bm25_size: int = 1000

    @property
    def data_path(self) -> str:
        dataset_settings = dataset_settings_cls[self.dataset]()
        return os.path.join(dataset_settings.data_path, str(dataset_settings.bm25_size))


@dataclass(kw_only=True)
class ZeroShot(Experiment):
    num_samples: int
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_ctx: Union[SentenceTransformer, None] = None
    scoring_fn: ScoringFunction = "cos"

    @property
    def model_class(self) -> str:
        if self.model.startswith("cross-encoder"):
            _model_class = "ce"
        else:
            _model_class = "bi"
        return _model_class


@dataclass(kw_only=True)
class KNNIndex(Experiment):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(kw_only=True)
class KNNSimilarities(Experiment):
    scoring_fn: ScoringFunction = "cos"


@dataclass(kw_only=True)
class QueryFineTune(Experiment):
    num_samples: int
    seed: int
    ft_params: FTParams = "bias"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    eval_batch_size: int = 32
    epochs: int = 8
    learning_rates: List[float] = field(default_factory=lambda: [2e-3, 2e-4, 2e-5])

    @property
    def model_class(self) -> str:
        if self.model.startswith("cross-encoder"):
            _model_class = "ce"
        else:
            _model_class = "bi"
        return _model_class

    @property
    def out_file(self) -> str:
        return os.path.join(
            self.data_path,
            f"k{self.num_samples}",
            f"s{self.seed}",
            f"valid_{self.model_class}_{self.ft_params}_few_shot_hpsearch.json",
        )
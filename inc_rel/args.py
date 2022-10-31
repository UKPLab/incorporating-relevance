import os
from enum import Enum
from dataclasses import dataclass, field, MISSING
from typing import List, Union

from settings import dataset_settings_cls
from sentence_transformers import SentenceTransformer


class ScoringFunction(Enum):
    COS = "cos"
    DOT = "dot"


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
    scoring_fn: ScoringFunction = ScoringFunction.COS

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
    scoring_fn: ScoringFunction = ScoringFunction.COS

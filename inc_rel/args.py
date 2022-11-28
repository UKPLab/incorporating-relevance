import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Union

from sentence_transformers import CrossEncoder, SentenceTransformer
from settings import dataset_settings_cls


class ScoringFunction(str, Enum):
    cos = "cos"
    dot = "dot"


class FTParams(str, Enum):
    full = "full"
    bias = "bias"
    # adapter = "adapter"
    head = "head"


class PreTrainMethod(str, Enum):
    supervised = "supervised"
    meta = "meta"


@dataclass(kw_only=True)
class Experiment:
    exp_name: str
    dataset: str
    num_samples: int
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    splits: List[str] = field(default_factory=lambda: ["train", "valid", "test"])
    bm25_size: int = 1000
    metric: str = "ndcg_cut_20"

    @property
    def dataset_settings(self):
        return dataset_settings_cls[self.dataset]()

    @property
    def data_path(self) -> str:
        return os.path.join(
            self.dataset_settings.data_path, str(self.dataset_settings.bm25_size)
        )

    @property
    def exp_path(self) -> str:
        _exp_path = os.path.join(
            self.dataset_settings.data_path, "experiments", self.exp_name
        )
        os.makedirs(_exp_path, exist_ok=True)
        return _exp_path

    @property
    def bm25_results(self) -> Dict:
        if not hasattr(self, "_bm25_results"):
            file = os.path.join(
                self.data_path, f"k{self.num_samples}", "expansion_results_16.json"
            )
            with open(file) as fh:
                self._bm25_results = json.load(fh)

        return self._bm25_results

    @property
    def bm25_docs(self) -> Dict:
        if not hasattr(self, "_bm25_docs"):
            file = os.path.join(self.data_path, "expansion_results_16.json")
            with open(file) as fh:
                self._bm25_docs = json.load(fh)

        return self._bm25_docs

    @property
    def qrels(self) -> Dict:
        if not hasattr(self, "_qrels"):
            file = os.path.join(self.data_path, "qrels.json")
            with open(file) as fh:
                self._qrels = json.load(fh)

        return self._qrels

    @property
    def topics(self) -> Dict:
        if not hasattr(self, "_topics"):
            file = os.path.join(self.data_path, "topics.json")
            with open(file) as fh:
                self._topics = json.load(fh)

        return self._topics

    @property
    def topic_ids_split_seed(self) -> Dict:
        if not hasattr(self, "_split_seed"):
            self._split_seed = {}
            for seed in self.seeds:
                for split in self.splits:
                    file = os.path.join(
                        self.data_path,
                        f"k{self.num_samples}",
                        f"s{seed}",
                        f"{split}.json",
                    )
                    with open(file) as fh:
                        self._split_seed[split, seed] = json.load(fh)

        return self._split_seed


@dataclass(kw_only=True)
class ZeroShot(Experiment):
    exp_name: str = "zero-shot"
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


class KNN(Experiment):
    exp_name: str = "knn"


@dataclass(kw_only=True)
class KNNIndex(KNN):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(kw_only=True)
class KNNSimilarities(KNN):
    scoring_fn: ScoringFunction = "cos"


@dataclass(kw_only=True)
class FineTuneExperiment(Experiment):
    exp_name: str = "query-ft"
    num_samples: int
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ft_params: FTParams = "bias"
    eval_batch_size: int = 32
    epochs: int = 8
    learning_rates: List[float] = field(default_factory=lambda: [2e-3, 2e-4, 2e-5])

    @property
    def model_class(self) -> str:
        if self.model.startswith("cross-encoder"):
            _model_class = CrossEncoder
        else:
            _model_class = SentenceTransformer
        return _model_class

    @property
    def hparam_results_file(self) -> str:
        return os.path.join(
            self.exp_path,
            f"k{self.num_samples}_s{{seed}}_valid_{self.ft_params}_hpsearch.json",
        )


@dataclass(kw_only=True)
class PreTrain(FineTuneExperiment):
    exp_name: str = "pt-query-ft"
    pretrain_method: PreTrainMethod = "meta"


@dataclass(kw_only=True)
class RankFusion(Experiment):
    exp_name: str = "rf"
    num_samples: int
    result_files: List[str]

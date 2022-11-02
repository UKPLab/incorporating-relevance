import time
from typing import Any, Dict, List

import numpy as np
import torch
from reranking_evaluator import RerankingEvaluator
from sentence_transformers import (
    CrossEncoder,
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data.dataloader import DataLoader

# from transformers import AdapterConfig


class FewShotTrainer:
    def __init__(
        self,
        model_name: str,
        ft_params: str,
        docs,
        inital_ranking,
        ranking_evaluator: RerankingEvaluator,
        pbar,
        model_path: str = None,
    ):

        self.model_name = model_name
        self.ft_params = ft_params
        self.docs = docs
        self.inital_ranking = inital_ranking
        self.ranking_evaluator = ranking_evaluator
        self.pbar = pbar
        self.model_path = model_path

    def init_model(self):
        if self.model_name.startswith("cross-encoder"):
            model = CrossEncoder(
                self.model_name if self.model_path is None else self.model_path
            )
            architecture = "ce"
        elif self.model_name.startswith("sentence-transformer"):
            model = SentenceTransformer(
                self.model_name if self.model_path is None else self.model_path
            )
            architecture = "bi"
        else:
            ValueError(self.model_name)

        return model, architecture

    def freeze_params(self):
        if self.architecture == "ce":
            np = self.model.model.named_parameters
        elif self.architecture == "bi":
            np = self.model.named_parameters
        else:
            raise ValueError(f"Unsupported Architecture: {self.architecture}")

        if self.ft_params == "head":
            if self.architecture == "bi":
                raise RuntimeError("Head Tuning not compatiable with Bi-Encoder.")
            for name, param in np():
                if "classifier" not in name:
                    param.requires_grad = False
        elif self.ft_params == "bias":
            for name, param in np():
                if "bias" not in name:
                    param.requires_grad = False
        elif self.ft_params == "full":
            pass
        # elif self.ft_params == "adapter":
        #     if self.architecture == "bi":
        #         model_ref = self.model[0].auto_model
        #     elif self.architecture == "ce":
        #         model_ref = self.model.model
        #     adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=16)
        #     adapter_name = "adapter"
        #     model_ref.add_adapter(adapter_name, config=adapter_config)

        #     model_ref.set_active_adapters("adapter")
        #     model_ref.train_adapter("adapter")
        else:
            raise ValueError(self.ft_params)

    def init_dataloader(self, annotations, batch_size):

        dataset = []
        for annotation in annotations:
            dataset.append(
                InputExample(
                    texts=[annotation["query"], annotation["document"]],
                    label=float(annotation["label"] > 0),
                )
            )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        topic2annotations,
        epochs: int,
        learning_rate: float,
        update_pbar: bool = True,
        iter_epochs: bool = True,
        time_it: bool = False,
    ) -> List[Dict[str, Any]]:
        results = []
        fit_times = []
        eval_times = []
        for topic_id, annotations in topic2annotations.items():

            # re-train the model for num_epochs each time from inital checkpoint
            for num_epochs in range(1, epochs + 1):
                if not iter_epochs:
                    if num_epochs != epochs:
                        continue

                torch.manual_seed(0)
                np.random.seed(0)

                self.model, self.architecture = self.init_model()
                self.freeze_params()

                t1 = time.time()
                self.fit(annotations, num_epochs, learning_rate)
                t2 = time.time()
                fit_times.append(t2 - t1)

                t1 = time.time()
                metrics, run = self.eval(annotations)
                t2 = time.time()
                eval_times.append(t2 - t1)

                results.append(
                    {
                        "topic_id": topic_id,
                        "epoch": num_epochs,
                        "learning_rate": learning_rate,
                        "metrics": metrics,
                        "run": run,
                    }
                )

                if update_pbar:
                    self.pbar.update(1)

        if time_it:
            return fit_times, eval_times

        return results

    def fit(self, annotations, num_epochs, learning_rate):
        dataloader = self.init_dataloader(annotations, batch_size=1)
        if self.architecture == "bi":
            loss = losses.CosineSimilarityLoss(self.model)
            train_args = {"train_objectives": [(dataloader, loss)]}
            self.model.train()
        elif self.architecture == "ce":
            train_args = {"train_dataloader": dataloader}
            self.model.model.train()
        else:
            raise ValueError(self.architecture)

        self.model.fit(
            **train_args,
            epochs=num_epochs,
            optimizer_params={"lr": learning_rate},
            save_best_model=False,
            show_progress_bar=False,
            warmup_steps=0,
        )

    def eval(self, annotations):

        return self.ranking_evaluator(
            model=self.model,
            queries={annotations[0]["topic_id"]: annotations[0]["query"]},
            docs=self.docs,
            inital_ranking=self.inital_ranking,
        )

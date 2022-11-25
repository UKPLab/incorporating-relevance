import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import eval
import numpy as np
import torch
import transformers
from meta.adapter_meta import extend_text_adapter
from meta.bias_meta import extend_linear
from meta.classification_head_meta import extend_classification_head
from meta.learner import Learner
from meta.meta_dataset import MetaDataset
from reranking_evaluator import RerankingEvaluator
from sentence_transformers import (
    CrossEncoder,
    InputExample,
    SentenceTransformer,
    losses,
    util,
)
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

# from transformers import AdapterConfig


class PreTrainTrainer:
    def __init__(
        self,
        model_name: str,
        ft_params: str,
        docs,
        inital_ranking,
        ranking_evaluator: RerankingEvaluator,
        meta,
        pbar,
    ):

        self.model_name = model_name
        self.ft_params = ft_params
        self.docs = docs
        self.inital_ranking = inital_ranking
        self.ranking_evaluator = ranking_evaluator
        self.meta = meta
        self.pbar = pbar

    def init_model(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_name.startswith("cross-encoder"):
            model = CrossEncoder(self.model_name)
            model_ref = model.model.bert
            named_parameters = model.model.named_parameters
            parameters = model.model.parameters
            loss_fn = BCEWithLogitsLoss()

        elif self.model_name.startswith("sentence-transformer"):
            model = SentenceTransformer(self.model_name)
            model.to(device)
            model_ref = model[0].auto_model
            named_parameters = model.named_parameters
            parameters = model.parameters
            loss_fn = losses.CosineSimilarityLoss(model)
        else:
            ValueError(self.model_name)

        # init params to fine-tune

        if self.ft_params == "bias":
            for name, param in named_parameters():
                if "bias" not in name:
                    param.requires_grad = False
            if self.meta:
                extend_linear(
                    model_ref,
                    prefix="bert."
                    if isinstance(model, CrossEncoder)
                    else "" + "encoder",
                )
                if isinstance(model, CrossEncoder):
                    model.model.classifier.weight.requires_grad = True
                    extend_classification_head(model.model)

        elif self.ft_params == "head":
            if isinstance(model, SentenceTransformer):
                raise RuntimeError("Head Tuning not compatiable with Bi-Encoder.")
            for name, param in named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            if self.meta:
                extend_classification_head(model_ref)
        # elif self.ft_params == "adapter":
        #     torch.manual_seed(0)
        #     adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=16)
        #     adapter_name = "adapter"
        #     model_ref.add_adapter(adapter_name, config=adapter_config)
        #     model_ref.set_active_adapters(adapter_name)
        #     model_ref.train_adapter(adapter_name)
        #     if self.meta:
        #         extend_text_adapter(model_ref)
        #         if isinstance(model, CrossEncoder):
        #             extend_classification_head(model.model)

        elif self.ft_params == "full":
            pass

        else:
            raise ValueError(self.ft_params)

        if self.model_name.startswith("cross-encoder"):
            model.model.to(device)
        elif self.model_name.startswith("sentence-transformer"):
            model.to(device)

        return model, loss_fn, parameters, named_parameters

    def init_optimizer(self, named_parameters, learning_rate):
        weight_decay = 0.01
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        return optimizer

    def init_dataloader(self, annotations, batch_size):
        dataset = []
        for annotation in annotations.values():
            for a in annotation:
                dataset.append(
                    InputExample(
                        texts=[a["query"], a["document"]],
                        label=float(a["label"] > 0),
                    )
                )
        g = torch.Generator()
        g.manual_seed(0)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    @staticmethod
    def batch_to_device(batch, device):
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def init_meta_dataloader(
        self, annotations, batch_size, tokenizer, device, architecture: str
    ):
        def collate_fn(batch):
            collated_batch = defaultdict(list)
            for sample in batch:
                for k, v in sample.items():
                    collated_batch[k].extend(v)
            collated_batch = dict(collated_batch)

            if architecture == "ce":
                query_doc_sequences = tokenizer(
                    collated_batch["query"],
                    collated_batch["documents"],
                    max_length=512,
                    truncation="only_second",
                    padding=True,
                    return_tensors="pt",
                )
                query_doc_sequences = self.batch_to_device(query_doc_sequences, device)
                padded_sequences = (query_doc_sequences, None)
            elif architecture == "bi":
                doc_sequences = tokenizer(
                    collated_batch["documents"],
                    max_length=512,
                    truncation="only_first",
                    padding=True,
                    return_tensors="pt",
                )
                doc_sequences = self.batch_to_device(doc_sequences, device)

                query_sequences = tokenizer(
                    collated_batch["query"][0],
                    max_length=512,
                    truncation="only_first",
                    padding=True,
                    return_tensors="pt",
                )
                query_sequences = self.batch_to_device(query_sequences, device)
                padded_sequences = (doc_sequences, query_sequences)
            else:
                raise ValueError(architecture)

            targets = torch.Tensor(collated_batch["labels"]).to(device)
            targets = (targets > 0).float()

            return padded_sequences, targets

        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(
            MetaDataset(annotations),
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            generator=g,
        )

    def train(self, *args, **kwargs):
        if self.meta:
            train_result = self.train_meta(*args, **kwargs)
        else:
            train_result = self.train_supvervised(*args, **kwargs)
        return train_result

    def train_supvervised(
        self,
        train_annotations: Dict[str, List],
        eval_annotations: Dict[str, List],
        epochs: int,
        learning_rate: float,
        selection_metric: str,
        exp_dir: str,
        max_grad_norm: float = 1,
        train_batch_size: int = 8,
        show_pbar: bool = True,
    ) -> List[Dict[str, Any]]:

        model, loss_fn, parameters, named_parameters = self.init_model()

        optimizer = self.init_optimizer(named_parameters, learning_rate)

        train_dataloader = self.init_dataloader(train_annotations, train_batch_size)
        train_dataloader.collate_fn = model.smart_batching_collate

        epoch_results = []
        best_epoch = 0
        best_metric = 0
        for epoch in range(epochs):

            for features, labels in train_dataloader:
                if isinstance(model, SentenceTransformer):
                    loss_value = loss_fn(features, labels)
                elif isinstance(model, CrossEncoder):
                    output = model.model(**features)
                    loss_value = loss_fn(output.logits.view(-1), labels)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            epoch_result, best_epoch, best_metric = self.eval(
                model,
                eval_annotations,
                epoch,
                learning_rate,
                best_epoch,
                best_metric,
                selection_metric,
                exp_dir,
            )
            epoch_results.append(epoch_result)
            self.pbar.update(1)

        return epoch_results, best_epoch, best_metric

    def train_meta(
        self,
        train_annotations: Dict[str, List],
        eval_annotations: Dict[str, List],
        epochs: int,
        learning_rate: float,
        selection_metric: str,
        exp_dir: str,
        max_grad_norm: float = 1,
        train_batch_size: int = 8,
        show_pbar: bool = True,
    ):
        model, loss_fn, parameters, named_parameters = self.init_model()
        architecture = "ce" if isinstance(model, CrossEncoder) else "bi"

        optimizer = self.init_optimizer(named_parameters, learning_rate)

        learner = Learner(
            base_model=model.model
            if isinstance(model, CrossEncoder)
            else model[0].auto_model,
            exclude_classifier=isinstance(model, SentenceTransformer),
            learning_rate=learning_rate,
            enable_warmup=False,
            num_warmup_steps=0,
            total_training_steps=None,
            num_steps_per_batch=1,
            meta_module=self.ft_params,
            loss_fn=torch.nn.BCEWithLogitsLoss
            if architecture == "ce"
            else torch.nn.MSELoss,
        )

        train_annotations, query_annotations = self.make_train_query_annotations(
            train_annotations
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader = self.init_meta_dataloader(
            annotations=train_annotations,
            batch_size=1,
            tokenizer=model.tokenizer,
            device=device,
            architecture=architecture,
        )
        query_dataloader = self.init_meta_dataloader(
            annotations=query_annotations,
            batch_size=1,
            tokenizer=model.tokenizer,
            device=device,
            architecture=architecture,
        )
        epoch_results = []
        best_epoch = 0
        best_metric = 0
        for epoch in range(epochs):
            for (train_batch, train_targets), (query_batch, query_targets) in zip(
                train_dataloader, query_dataloader
            ):
                params, results = learner.train_step(
                    **train_batch[0],
                    targets=train_targets,
                    batch_size=train_batch_size,
                    eval_final=True,
                    architecture=architecture,
                    query_batch=train_batch[1],
                )
                if architecture == "ce":
                    logits = learner(**query_batch[0], params=params,)[
                        0
                    ].view(-1)
                    loss = loss_fn(logits, query_targets)
                elif architecture == "bi":
                    doc_out = learner(**query_batch[0], params=params)
                    doc_embeddings = doc_out[1]
                    query_out = learner(**query_batch[1], params=params)
                    query_embedding = query_out[1]
                    logits = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
                    loss = torch.nn.MSELoss()(logits, query_targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters(), max_grad_norm)
                optimizer.step()

            epoch_result, best_epoch, best_metric = self.eval(
                model,
                eval_annotations,
                epoch,
                learning_rate,
                best_epoch,
                best_metric,
                selection_metric,
                exp_dir,
            )
            epoch_results.append(epoch_result)
            self.pbar.update(1)

        return epoch_results, best_epoch, best_metric

    def eval(
        self,
        model,
        annotations,
        epoch,
        learning_rate,
        best_epoch,
        best_metric,
        selection_metric,
        exp_dir,
    ):

        queries = {
            topic_id: annotation[0]["query"]  # query is the same in each annotaiton
            for topic_id, annotation in annotations.items()
        }
        metrics, run = self.ranking_evaluator(
            model=model,
            queries=queries,
            docs=self.docs,
            inital_ranking=self.inital_ranking,
        )

        if eval.accumulate_results(metrics)["mean"][selection_metric] > best_metric:
            best_metric = eval.accumulate_results(metrics)["mean"][selection_metric]
            best_epoch = epoch
            os.makedirs(exp_dir, exist_ok=True)
            model.save(os.path.join(exp_dir, "model"))

        result = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "metrics": metrics,
            "run": run,
        }
        if best_epoch == epoch:
            with open(
                os.path.join(exp_dir, "valid_best_epoch_results.json"), "w"
            ) as fh:
                json.dump(result, fh, indent=4)

        return result, best_epoch, best_metric

    @staticmethod
    def make_train_query_annotations(annotations, seed=1):
        topic_ids = list(annotations.keys())
        random.seed(seed)
        random.shuffle(topic_ids)
        n = len(topic_ids) // 2
        train_topic_ids, query_topic_ids = topic_ids[:n], topic_ids[-n:]
        train_annotations = {
            topic_id: annotations[topic_id] for topic_id in train_topic_ids
        }
        query_annotations = {
            topic_id: annotations[topic_id] for topic_id in query_topic_ids
        }
        return train_annotations, query_annotations

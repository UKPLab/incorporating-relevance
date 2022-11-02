import logging
import math
from collections import defaultdict
from typing import Dict, Union

import meta.utils as utils
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import util


class Learner(nn.Module):
    def __init__(
        self,
        base_model,
        exclude_classifier: bool,
        learning_rate: float,
        enable_warmup: bool,
        num_warmup_steps: int,
        total_training_steps: int,
        num_steps_per_batch: int,
        meta_module: str,
        loss_fn,
    ):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = loss_fn()
        self.exclude_classifier = exclude_classifier
        self.learning_rate = learning_rate
        self.enable_warmup = enable_warmup
        self.meta_module = meta_module
        self.num_steps_per_batch = num_steps_per_batch
        if num_warmup_steps > 0:
            self.lr_scheduler = utils.InnerLRScheduler(
                learning_rate, num_warmup_steps, total_training_steps
            )
        # TODO make adapter name variable

    def meta_module_iter(self, get_all: bool = False):

        if self.meta_module == "adapter":
            for layer in getattr(
                self.base_model, "bert", self.base_model
            ).encoder.layer:
                for adapter in layer.output.adapters.values():
                    yield adapter
        elif self.meta_module == "bias":
            for n, m in getattr(
                self.base_model, "bert", self.base_model
            ).encoder.named_modules(
                prefix="bert." if hasattr(self.base_model, "bert") else "" + "encoder"
            ):
                if isinstance(m, nn.Linear):
                    assert m.bias.requires_grad, n
                    yield m

        if get_all or not self.exclude_classifier:
            assert self.base_model.classifier.weight.requires_grad
            assert self.base_model.classifier.bias.requires_grad
            yield self.base_model.classifier

    def get_params(self):
        params = {}
        for meta_module in self.meta_module_iter():
            params.update(meta_module.params)

        return params

    def set_params(self, params: Dict[str, torch.Tensor]):
        for meta_module in self.meta_module_iter():
            meta_module.set_params(params)

    def reset_params(self):
        for meta_module in self.meta_module_iter(get_all=False):
            meta_module.reset_params()

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        params: Union[Dict[str, torch.Tensor], None] = None,
    ):
        if params:
            self.set_params(params)
        else:
            self.reset_params()

        out = self.base_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        # out = out[0].view(-1)

        return out

    def train_step(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        targets,
        batch_size,
        eval_final: bool,
        architecture: str,
        query_batch: Dict = None,
    ):
        results = defaultdict(list)
        params = None

        self.reset_params()
        for batch_i, batch_inputs, batch_targets in utils.batch_iter(
            math.ceil(input_ids.size(0) / self.num_steps_per_batch),
            input_ids,
            token_type_ids,
            attention_mask,
            targets,
            shuffle=True,
        ):
            if architecture == "ce":
                out = self(*batch_inputs, params=params)
                logits = out[0].view(-1)

            elif architecture == "bi":
                doc_out = self(*batch_inputs, params=params)
                doc_embeddings = doc_out[1]
                query_out = self(**query_batch, params=params)
                query_embedding = query_out[1]

                logits = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            else:
                raise ValueError(architecture)

            loss = self.loss_fn(logits, batch_targets)

            if not params:
                params = self.get_params()
            self.base_model.zero_grad()
            try:
                grads = torch.autograd.grad(
                    loss, params.values(), create_graph=False, allow_unused=False
                )

            except Exception as err:
                print(params.keys())
                grads = torch.autograd.grad(
                    loss, params.values(), create_graph=False, allow_unused=True
                )
                for (name, param), grad in zip(params.items(), grads):
                    if grad is None or grad.sum() == 0:
                        print("grad is zero for ", name)
                print(params.keys())
                raise err

            lr = (
                self.lr_scheduler.get_lr() if self.enable_warmup else self.learning_rate
            )
            updated_params = {}
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - lr * grad

            if self.enable_warmup:
                self.lr_scheduler.step()

            params = updated_params

            with torch.no_grad():
                results["loss"].append(loss.item())
                accuracy = utils.binary_accuracy_from_logits(batch_targets, logits)
                results["accuracy"].append(accuracy)

        if eval_final:
            with torch.no_grad():
                logits = self(
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    params=params,
                )
                if architecture == "ce":
                    out = self(
                        input_ids,
                        token_type_ids,
                        attention_mask,
                        params=params,
                    )
                    logits = out[0].view(-1)

                elif architecture == "bi":
                    doc_out = self(
                        input_ids, token_type_ids, attention_mask, params=params
                    )
                    doc_embeddings = doc_out[1]
                    query_out = self(**query_batch, params=params)
                    query_embedding = query_out[1]

                    logits = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            loss = self.loss_fn(logits, targets)
            accuracy = utils.binary_accuracy_from_logits(targets, logits)
            results["final-loss"].append(loss.item())
            results["final-accuracy"].append(accuracy)

        for k, v in results.items():
            if isinstance(v, list):
                results[k] = np.mean(v)

        return params, results

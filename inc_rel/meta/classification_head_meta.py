import logging
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from meta.module_meta import ModuleMetaMixin

logger = logging.getLogger(__name__)


class ClassificationHeadMixin(ModuleMetaMixin):
    def get_params(self):
        params = {
            "clf.linear.weight": self.weight,
            "clf.linear.bias": self.bias,
        }
        return params

    def set_params(self, params):
        params = dict(filter(lambda kv: f"clf" in kv[0], params.items()))
        self.params = params

    def forward(self, input):

        return F.linear(
            input=input,
            weight=self.params["clf.linear.weight"],
            bias=self.params["clf.linear.bias"],
        )


def extend_classification_head(model):
    base_cls = model.classifier.__class__
    base_cls_name = model.classifier.__class__.__name__
    model.classifier.__class__ = type(
        base_cls_name, (ClassificationHeadMixin, base_cls), {}
    )

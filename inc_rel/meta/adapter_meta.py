import logging
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from meta.module_meta import ModuleMetaMixin

logger = logging.getLogger(__name__)


class AdapterMetaMixin(ModuleMetaMixin):
    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        self._layer = layer

    def get_params(self):
        return {
            f"layer.{self.layer}.adapter_down.weight": self.adapter_down[0].weight,
            f"layer.{self.layer}.adapter_down.bias": self.adapter_down[0].bias,
            f"layer.{self.layer}.adapter_up.weight": self.adapter_up.weight,
            f"layer.{self.layer}.adapter_up.bias": self.adapter_up.bias,
        }

    def set_params(self, params):
        params = dict(filter(lambda kv: f"layer.{self.layer}" in kv[0], params.items()))
        self.params = params

    def forward(self, x, residual_input):
        """forward pass with functional functions for meta learning"""

        down = F.linear(
            input=x,
            weight=self.params[f"layer.{self.layer}.adapter_down.weight"],
            bias=self.params[f"layer.{self.layer}.adapter_down.bias"],
        )
        down = self.adapter_down[1](down)  # activation function

        up = F.linear(
            input=down,
            weight=self.params[f"layer.{self.layer}.adapter_up.weight"],
            bias=self.params[f"layer.{self.layer}.adapter_up.bias"],
        )

        output = up

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(
                output
            )  # this can stay since its not affected by meta learning

        # if residual should be applied after layer norm, apply it here
        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up


def extend_text_adapter(model):
    for layer_i, layer in enumerate(model.encoder.layer):
        for adapter_obj in layer.output.adapters.values():
            # https://stackoverflow.com/a/31075641
            base_cls = adapter_obj.__class__
            base_cls_name = adapter_obj.__class__.__name__
            adapter_obj.__class__ = type(
                base_cls_name, (AdapterMetaMixin, base_cls), {}
            )
            adapter_obj.layer = layer_i

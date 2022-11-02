import torch.nn as nn
import torch.nn.functional as F
from meta.module_meta import ModuleMetaMixin


class LinearBiasMetaMixin(ModuleMetaMixin):
    @property
    def module_name(self):
        return self._module_name

    @module_name.setter
    def module_name(self, module_name):
        self._module_name = module_name

    def get_params(self):
        return {self.module_name: self.bias}

    def set_params(self, params):
        self.params = {self.module_name: params[self.module_name]}

    def forward(self, x):
        """forward pass with functional functions for meta learning"""

        return F.linear(
            input=x,
            weight=self.weight,
            bias=self.params[self.module_name],
        )


def extend_linear(model, prefix=""):

    for n, m in model.encoder.named_modules(prefix=prefix):
        if isinstance(m, nn.Linear):
            base_cls = m.__class__
            m.__class__ = type("MetaLinear", (LinearBiasMetaMixin, base_cls), {})
            m.module_name = n + ".bias"

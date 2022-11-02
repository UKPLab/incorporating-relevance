from typing import Dict

import torch.nn as nn


class ModuleMetaMixin(nn.Module):
    def get_params(self):
        raise NotImplementedError()

    def set_params(self, params):
        self._params = params

    def reset_params(self):
        self.params = self.get_params()

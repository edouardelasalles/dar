import warnings

import torch.nn as nn
import torch.nn.functional as F


class LockedDropout(nn.Module):
    '''
    Code borrowed from AWD-LSTM :
    https://github.com/salesforce/awd-lstm-lm/
    '''
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout <= 0:
            return x
        m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = m / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class WeightDropout(nn.Module):
    '''
    Code borrowed from fastai:
    https://github.com/fastai/fastai/blob/9336a188c2a0362fc64a33185daa1779a9bf035b/fastai/text/models/awd_lstm.py#L26
    A module that warps another layer in which some weights will be replaced by 0 during training.
    '''

    def __init__(self, module, weight_p, layer_names=['weight_hh_l0']):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'):
            self.module.reset()

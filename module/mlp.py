import torch.nn as nn


def make_lin_block(ninp, nout, bias, relu, dropout):
    modules = []
    if dropout > 0:
        modules.append(nn.Dropout(dropout))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    modules.append(nn.Linear(ninp, nout, bias=bias))
    return nn.Sequential(*modules)


class MLP(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, bias=True, dropout=0.):
        super().__init__()
        assert nhid == 0 or nlayers > 1
        assert dropout == 0 or nlayers > 1
        # attributes
        self.ninp = ninp
        self.nout = nout
        # modules
        modules = [
            make_lin_block(
                ninp=ninp if il == 0 else nhid,
                nout=nout if il == nlayers - 1 else nhid,
                bias=bias,
                relu=il > 0,
                dropout=dropout if il > 0 else 0.,
            ) for il in range(nlayers)
        ]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)

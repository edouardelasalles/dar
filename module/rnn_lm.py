import torch
import torch.nn as nn

from module.dropout import LockedDropout, WeightDropout


class RNNLanguageModel(nn.Module):
    def __init__(self, ntoken, nwe, nz, cond_fusion, nhid, nout, nlayers, cell, dropouti, dropoutl, dropoutw, dropouto):
        super().__init__()
        assert cond_fusion in ('cat', 'h0', 'w0')
        assert dropoutl == 0 or nlayers > 1
        # attributes
        self.do_downproj = nlayers == 1 and nout != nhid
        self.nhid = nhid
        self.nlayers = nlayers
        self.cond_fusion = cond_fusion
        self.cell = cell
        self.nwe = nwe
        self.nz = nz
        # modules
        # -- rnn
        ninp = nwe + nz if cond_fusion == 'cat' else nwe
        nout = nhid if self.do_downproj else nout
        self.rnn = RNN(cell, ninp, nhid, nout, nlayers, dropouti, dropoutl, dropoutw, dropouto)
        # -- condition fusion
        if cond_fusion == 'h0':
            self.up_proj = nn.Linear(nz, nhid * nlayers)
        if cond_fusion == 'w0':
            self.up_proj = nn.Linear(nz, nwe)
        # -- down projection
        if self.do_downproj:
            self.downproj = nn.linear(nhid, nout)
        # -- decoder
        self.decoder = nn.Linear(nout, ntoken)
        self.decoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()

    def tie_weights(self, weight):
        self.decoder.weight = weight

    def forward(self, z, emb, hidden=None):
        inputs, hidden = self._get_inputs(z, emb, hidden)
        output, hidden = self.rnn(inputs, hidden)
        output = self.decode(output)
        return output, hidden

    def decode(self, output):
        if self.do_downproj:
            output = self.downproj(output)
        return self.decoder(output)

    def _get_inputs(self, z, emb, hidden):
        if self.cond_fusion == 'cat':
            z_ex = z.unsqueeze(0).expand(emb.shape[0], *z.shape)
            inputs = torch.cat([emb, z_ex], 2)
            return inputs, hidden
        if hidden is None:
            if self.cond_fusion == 'h0':
                z_proj = self.up_proj(z)
                h0 = z_proj.view(-1, self.nlayers, self.nhid).transpose(0, 1).contiguous()
                hidden = [h.unsqueeze(0) for h in h0]
                if self.cell == 'LSTM':
                    c0 = [torch.zeros_like(h) for h in hidden]
                    hidden = (hidden, c0)
                return emb, hidden
            if self.cond_fusion == 'w0':
                emb[0] = self.up_proj(z)
                return emb, None
        # if hidden is not None, then we are not at first token
        return emb, hidden


class RNN(nn.Module):
    '''
    Code inspired by AWD-LSTM :
    Regularizing and Optimizing LSTM Language Models.
    Merity et al. ICLR 2018
    https://github.com/salesforce/awd-lstm-lm/
    '''
    def __init__(self, cell, ninp, nhid, nout, nlayers, dropouti, dropoutl, dropoutw, dropouto):
        super(RNN, self).__init__()
        assert nlayers > 1 or dropoutl == 0.0, 'Layer dropout only in multi layers lstm'
        assert nlayers > 1 or nout == nhid, 'if one layer, nhid = nwe'
        # attributes
        self.cell = cell
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.nlayers = nlayers
        self.idrop = LockedDropout(dropouti)
        self.ldrop = LockedDropout(dropoutl)
        self.odrop = LockedDropout(dropouto)
        # LSTM
        self.rnns = [getattr(nn, self.cell)(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nout, 1)
                     for l in range(nlayers)]
        if dropoutw > 0:
            self.rnns = [WeightDropout(rnn, dropoutw) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)

    def forward(self, input, hidden=None):
        output = input
        new_hidden = []
        outputs = []
        raw_outputs = []
        # forward through layers
        for l, rnn in enumerate(self.rnns):
            input_l = output
            if hidden is None:
                h_n = None
            elif self.cell == 'LSTM':
                h_n = (hidden[0][l], hidden[1][l])
            else:
                h_n = hidden[l]
            # dropout
            input_l_droped = self.idrop(input_l) if l == 0 else self.ldrop(input_l)
            if l > 0:
                outputs.append(input_l_droped)
            # forward
            raw_output, new_h = rnn(input_l_droped, h_n)
            raw_outputs.append(raw_output)
            new_hidden.append(new_h)
            output = raw_output
        raw_output = output
        output = self.odrop(output)
        outputs.append(output)
        if self.cell == 'LSTM':
            h_n = [h_n_l for h_n_l, _ in new_hidden]
            c_n = [c_n_l for _, c_n_l in new_hidden]
            hidden = (h_n, c_n)
        else:
            hidden = new_hidden
        return output, hidden

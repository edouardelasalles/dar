from functools import partial

import torch
import torch.nn as nn

from module.rnn_lm import RNNLanguageModel
from module.mlp import MLP
from module.utils import init_weight


class DynamicAuthorLanguageModel(nn.Module):
    def __init__(self,
                 ntoken, nwe,  # text
                 naut, nha, nhat, nhid_dyn, nlayers_dyn, cond_fusion,  # authors
                 nhid_lm, nlayers_lm, dropouti, dropoutl, dropoutw, dropouto, tie_weights, padding_idx,  # language model
                 ):
        super().__init__()
        # attributes
        self.ntoken = ntoken  # size of vocabulary
        self.nwe = nwe  # size of word embeddings
        self.naut = naut  # number of authors
        self.nha = nha  # size of static author embeddings
        self.nhat = nhat  # size of dynamic latent states
        self.tie_weights = tie_weights  # tie weights between word embeddings and linear decoder?
        self.padding_idx = padding_idx
        self.cond_fusion = cond_fusion  # how to incorporate the context vectors into the language mode? (w0|h0|cat)
        nout = nwe if tie_weights else nhid_lm
        # modules
        # -- word embeddings
        self.word_embedding = nn.Embedding(ntoken, nwe, padding_idx=padding_idx)
        # -- static author embeddings
        self.author_embedding = nn.Embedding(naut, nha, padding_idx=padding_idx)
        # -- LSTM language model
        self.rnn_lm = RNNLanguageModel(ntoken, nwe, nha + nhat, self.cond_fusion, nhid_lm, nout, nlayers_lm, 'LSTM',
                                       dropouti, dropoutl, dropoutw, dropouto)
        # dynamic modules
        nhid_init = 0 if nlayers_dyn == 1 else nhid_dyn
        # -- mlp that that produce h_{a,t=0} from static author embeddings
        self.init_dyn = MLP(nha, nhid_init, nhat, nlayers_dyn)
        # -- mlp for the dynamic residual latent function
        self.dynamic = MLP(nha + nhat, nhid_dyn, nhat, nlayers_dyn)
        # init
        self._init()

    def _init(self):
        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.word_embedding.weight.data[self.padding_idx] = 0
        self.author_embedding.weight.data.uniform_(-0.1, 0.1)
        self.author_embedding.weight.data[self.padding_idx] = 0
        # weight tying
        if self.tie_weights:
            self.rnn_lm.tie_weights(self.word_embedding.weight)
        # we initialise the residual MLP orthogonally to increase the numerical stability of latent
        # states through time
        init_weight_fn = partial(init_weight, init_type='orthogonal', init_gain=0.02)
        self.dynamic.apply(init_weight_fn)

    def init_state(self, ha):
        # compute the initial state h_{a,t=0}
        ha0 = self.init_dyn(ha)
        return ha0

    def next_state(self, state, ha):
        # compute the next state given the previous one and the static embedding
        res_inp = torch.cat([state, ha], 1)
        res = self.dynamic(res_inp)
        next_state = state + res
        return next_state, res

    def get_cond(self, authors, timesteps):
        # compute the context vectors for a batch of (author, timestep) pairs
        # authors: LongTensor with shape [batch_size] containing author ids
        # timesteps: LongTensor with shape [batch_size] containing timesteps or a scalar timestep
        nt = timesteps + 1 if isinstance(timesteps, int) else timesteps.max().item() + 1
        # -- get the static author embeddings and the initial state
        ha = self.author_embedding(authors)
        hat = [self.init_state(ha)]
        res = []
        # -- loop through time to compute all context vectors
        for t in range(1, nt):
            hat_next, res_t = self.next_state(hat[-1], ha)
            hat.append(hat_next)
            res.append(res_t)
        hat_all_t = torch.stack(hat)
        # -- retrive the context vectors corresponding to the (author, timestep) pairs given in input
        if isinstance(timesteps, int):
            hat = hat_all_t[timesteps]
        else:
            i_range = torch.arange(len(timesteps), device=timesteps.device)
            hat = hat_all_t[timesteps, i_range]
        return ha, hat

    def decode(self, emb, ha, hat, hidden=None):
        # the final context vector if formed by concatenating a static embeddings ha and a dynamic state hat
        cond = torch.cat([ha, hat], 1)
        # the final context vector is fed to the LSTM language model
        return self.rnn_lm(cond, emb, hidden=hidden)

    def forward(self, text, authors, timesteps):
        # text: LongTensor of token indices with shape [seq_len, batch_size]
        # authors: LongTensor of author ods with shape [batch_size]
        # timesteps: LongTensor of timesteps with shape [batch_size] or a scalar timestep
        ha, hat = self.get_cond(authors, timesteps)
        emb = self.word_embedding(text)
        output, hidden = self.decode(emb, ha, hat)
        return output, hidden

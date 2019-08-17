import os
import math

import torch.nn.functional as F

from corpus import BertCorpus, Corpus


def neg_log_prob(proba, data):
    nll = F.cross_entropy(proba.view(-1, proba.shape[-1]), data.view(-1), ignore_index=Corpus.pad_id, reduction='none')
    return nll.view_as(data)


def perplexity(nll):
    try:
        return math.exp(nll)
    except OverflowError:
        return float('inf')


def load_corpus(opt):
    opt.data_dir = os.path.join(opt.dataroot, opt.corpus, opt.task)
    if opt.corpus == 's2':
        return BertCorpus.load_corpus(opt.data_dir, bert_cache_dir=opt.bert_cache_dir)
    if opt.corpus == 'nyt':
        return Corpus.load_corpus(opt.data_dir)


def load_fold(corpus, fold, data_dir):
    with open(os.path.join(data_dir, f'{fold}.txt'), 'r') as f:
        ids = set(f.read().splitlines())
    corpus = corpus.filter_ids(ids)
    print(f'{fold} set: {len(corpus)} examples ({corpus.nwords} words), {corpus.na} authors, {corpus.nt} timesteps')
    return corpus

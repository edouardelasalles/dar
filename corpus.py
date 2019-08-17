import os
import re
import json
from collections import Counter, namedtuple, defaultdict

import pickle
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence

import nltk
from pytorch_pretrained_bert import BertTokenizer


Element = namedtuple('Element', ['id', 'text', 'author', 'timestep'])


class Corpus(object):
    bos_token = '<bos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    unk_token = '<unk>'

    def __init__(self, examples, ids, vocab, authors):
        self.examples = examples
        self.ids = ids
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.nwords = sum(len(ex.text) for ex in self.examples)
        self.authors = authors
        self.na = len(authors.i2s)
        self.nt = len(set([ex.timestep for ex in examples]))
        self._init_specials()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        x = self.examples[index]
        text = torch.LongTensor(self.vocab.convert_tokens_to_ids(x.text))
        return Element(x.id, text, self.authors.s2i[x.author], x.timestep)

    def post_process(self, text):
        return text

    def filter_ids(self, ids):
        examples = list(filter(lambda x: x.id in ids, self.examples))
        return self.__class__(examples, self.ids, self.vocab, self.authors)

    def _init_specials(self):
        specials = [Corpus.pad_token, Corpus.unk_token, Corpus.bos_token, Corpus.eos_token]
        Corpus.pad_id, Corpus.unk_id, Corpus.bos_id, Corpus.eos_id = self.vocab.convert_tokens_to_ids(specials)
        assert Corpus.unk_token in self.vocab.i2s and Corpus.unk_id == 1

    @classmethod
    def load_corpus(cls, data_dir, **kwargs):
        pckl_path = os.path.join(data_dir, 'corpus.pkl')
        if os.path.isfile(pckl_path):
            print(f'Loading corpus at {pckl_path}...')
            with open(pckl_path, 'rb') as f:
                return cls(*pickle.load(f))
        # corpus
        fpath = os.path.join(data_dir, 'corpus.json')
        print(f'Loading corpus at {fpath}...')
        data = {}
        with open(fpath, 'r') as f:
            for l in f.read().splitlines():
                ex = json.loads(l)
                data[ex['id']] = ex
        # fields
        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            train_ids = set(f.read().splitlines())
        trainset = list(filter(lambda x: x['id'] in train_ids, data.values()))
        # -- ids
        ids = Vocab(list(data.keys()))
        # -- texts
        vocab = cls._build_vocab(trainset, **kwargs)
        # -- authors
        authors = Vocab([e for ex in trainset for e in ex['authors']], specials=['<pad>'], unk_token=False)
        # preprocess
        print('preprocessing...')
        examples = cls._preprocess(data, ids.i2s, vocab, authors)
        with open(pckl_path, 'wb') as f:
            pickle.dump((examples, ids, vocab, authors), f)
        return cls(examples, ids, vocab, authors)

    @classmethod
    def _build_vocab(cls, data):
        all_tokens = []
        for ex in data:
            text = ' '.join(text.strip().lower() for text in ex['texts'])
            all_tokens += cls._tokenize(text)
        specials = [cls.pad_token, cls.unk_token, cls.bos_token, cls.eos_token]
        vocab = Vocab(all_tokens, min_freq=5, specials=specials)
        return vocab

    @classmethod
    def _preprocess(cls, data, ids, text_vocab, authors_vocab):
        # tokenize
        examples = []
        for i in tqdm(ids):
            ex = data[i]
            t = ex['timestep']
            authors = ex['authors']
            tokens = cls._preprocess_texts(ex['texts'], text_vocab)
            if len(tokens) > 512:
                continue
            for a in authors:
                examples.append(Element(i, tokens, a, t))
        return examples

    @classmethod
    def _tokenize(self, text):
        return ['N' if re.search(r'[0-9]', w) else w for w in nltk.tokenize.word_tokenize(text)]

    @classmethod
    def _preprocess_texts(cls, texts, vocab):
        text = ' '.join([text.strip().lower() for text in texts])
        tokens = [cls.bos_token] + cls._tokenize(text) + [cls.eos_token]
        return tokens


class BertCorpus(Corpus):
    pad_token = '[PAD]'
    bos_token = '[CLS]'
    eos_token = '[SEP]'
    unk_token = '[UNK]'

    def _init_wp(self, vocab):
        self.is_full_word = torch.tensor([not wp.startswith('##') for wp in vocab.vocab.keys()])

    def _init_specials(self):
        specials = [BertCorpus.bos_token, BertCorpus.eos_token, BertCorpus.pad_token, BertCorpus.unk_token]
        Corpus.bos_token, Corpus.eos_token, Corpus.pad_token, Corpus.unk_token = specials
        Corpus.bos_id, Corpus.eos_id, Corpus.pad_id, Corpus.unk_id = self.vocab.convert_tokens_to_ids(specials)

    def post_process(self, text):
        out = ''
        for word in text:
            if word[:2] == '##':
                out += word[2:]
            else:
                out += word
        return out

    @classmethod
    def _build_vocab(cls, data, bert_cache_dir):
        print(f'(Down)Loading BERT tokenizer at {bert_cache_dir}')
        vocab = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=bert_cache_dir)
        vocab.size = len(vocab.vocab)
        return vocab

    @classmethod
    def _preprocess_texts(cls, texts, vocab):
        text = cls.bos_token + ' ' + ' '.join(texts) + ' ' + cls.eos_token
        tokens = vocab.tokenize(text)
        return tokens


class Vocab():
    default_id = None

    def __init__(self, data, min_freq=0, specials=None, unk_token=True):
        self.min_freq = min_freq
        counter = Counter(data)
        self.frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        self.frequencies.sort(key=lambda tup: tup[1], reverse=True)
        self.specials = specials
        self.i2s = [w for w in self.specials] if self.specials is not None else []
        for word, freq in self.frequencies:
            if freq < self.min_freq:
                break
            self.i2s.append(word)
        if unk_token:
            self.s2i = defaultdict(_default_id)
        else:
            self.s2i = {}
        self.s2i.update({tok: i for i, tok in enumerate(self.i2s)})
        self.size = len(self.i2s)

    def convert_tokens_to_ids(self, tokens):
        return [self.s2i[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.i2s[i] for i in ids]

    def __len__(self):
        return len(self.i2s)


def text_collate(batch):
    elements = Element(*zip(*batch))
    texts = pad_sequence(elements.text, batch_first=False, padding_value=Corpus.pad_id)
    authors = torch.tensor(elements.author)
    timesteps = torch.tensor(elements.timestep)
    return Element(elements.id, texts, authors, timesteps)


def _default_id():
    return 1

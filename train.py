import os
import shutil
import random
import argparse

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model import DynamicAuthorLanguageModel
from corpus import Corpus, text_collate
from utils import load_corpus, load_fold, neg_log_prob, perplexity


def train_step(model, optimizer, batch, device, opt):
    # perform a single stochastic gradient step
    model.train()
    optimizer.zero_grad()
    # extract data from batch and send them to GPU
    text = batch.text.to(device)
    input_text = text[:-1]
    target_text = text[1:]
    authors = batch.author.to(device)
    timesteps = batch.timestep.to(device)
    n = text.shape[1]
    ntkn = target_text.ne(Corpus.pad_id).sum().item()
    # forward
    pred, _ = model(input_text, authors, timesteps)
    # loss
    loss = 0
    # -- word level nll
    nll = neg_log_prob(pred, target_text).sum() / n
    loss += nll
    # -- L2 regularization of static word embeddings
    if opt.l2_a > 0:
        ha = model.author_embedding.weight
        loss += opt.l2_a * 0.5 * ha.pow(2).sum() / opt.n_ex
    # backward
    loss.backward()
    # step
    optimizer.step()
    return perplexity(nll.item() * n / ntkn)


def evaluate(model, testloader, device):
    model.eval()
    ntkn = 0
    nll = 0
    for batch in testloader:
        # data
        text = batch.text.to(device)
        input_text = text[:-1]
        target_text = text[1:]
        authors = batch.author.to(device)
        timesteps = batch.timestep.to(device)
        ntkn += target_text.ne(Corpus.pad_id).sum().item()
        # forward
        pred, _ = model(input_text, authors, timesteps)
        # perplexity
        nll += neg_log_prob(pred, target_text).sum().item()
    return perplexity(nll / ntkn)


def main(opt):
    opt.hostname = os.uname()[1]
    # cudnn
    if opt.device.lstrip('-').isdigit() and int(opt.device) <= -1:
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
        device = torch.device('cuda')
    # seed
    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)
    print(f"seed: {opt.manual_seed}")
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    # xp dir
    if os.path.isdir(opt.xp_dir):
        if input(f'Experiment folder already exists at {opt.xp_dir}. Erase it? (y|n)') in ('yes', 'y'):
            shutil.rmtree(opt.xp_dir)
        else:
            print('Terminating experiment...')
            exit(0)
    os.makedirs(opt.xp_dir)
    print(f'Experiment directory created at {opt.xp_dir}')

    ##################################################################################################################
    # Data
    ##################################################################################################################
    print('Loading data...')
    # load corpus
    corpus = load_corpus(opt)
    # trainset
    trainset = load_fold(corpus, 'train', opt.data_dir)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, collate_fn=text_collate, shuffle=True,
                             pin_memory=True, drop_last=True)
    # testset
    testset = load_fold(corpus, 'test', opt.data_dir)
    testloader = DataLoader(testset, batch_size=opt.batch_size, collate_fn=text_collate, shuffle=False,
                            pin_memory=True)
    # attributes
    opt.n_ex = len(trainset)
    opt.naut = trainset.na
    opt.ntoken = corpus.vocab_size
    opt.padding_idx = Corpus.pad_id

    ##################################################################################################################
    # Model
    ##################################################################################################################
    print('Building model...')
    model = DynamicAuthorLanguageModel(opt.ntoken, opt.nwe, opt.naut, opt.nha, opt.nhat, opt.nhid_dyn, opt.nlayers_dyn,
                                       opt.cond_fusion, opt.nhid_lm, opt.nlayers_lm, opt.dropouti, opt.dropoutl,
                                       opt.dropoutw, opt.dropouto, opt.tie_weights, opt.padding_idx).to(device)
    opt.model = str(model)
    opt.nparameters = sum(p.nelement() for p in model.parameters())
    print(f'{opt.nparameters} parameters')

    ##################################################################################################################
    # Optimizer
    ##################################################################################################################
    model_params = list(model.named_parameters())
    no_wd = ['entity_embedding']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_wd)], 'weight_decay': opt.wd},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_wd)], 'weight_decay': 0.0}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=opt.lr)
    opt.optimizer = str(optimizer)
    # learning rate scheduling
    niter = opt.lr_scheduling_burnin + opt.lr_scheduling_niter
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda i: max(0, (opt.lr_scheduling_niter - i) / opt.lr_scheduling_niter))

    ##################################################################################################################
    # Training
    ##################################################################################################################
    print('Training...')
    cudnn.benchmark = True
    assert niter > 0
    pb = tqdm(total=niter, ncols=0, desc='iter')
    itr = -1
    finished = False
    ppl_test = None
    while not finished:
        # train
        for batch in trainloader:
            itr += 1
            # gradient step
            ppl_train = train_step(model, optimizer, batch, device, opt)
            # lr scheduling
            if itr >= opt.lr_scheduling_burnin:
                lr_scheduler.step()
            # progress bar
            pb.set_postfix(ppl_train=ppl_train, ppl_test=ppl_test, lr=optimizer.param_groups[0]['lr'])
            pb.update()
            # break ?
            if itr > 0 and itr % opt.chkpt_interval == 0:
                break
            if itr >= niter:
                finished = True
                break
        # eval
        if itr % opt.chkpt_interval == 0:
            with torch.no_grad():
                ppl_test = evaluate(model, testloader, device)
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'opt': opt},
                os.path.join(opt.xp_dir, 'model.pth')
            )
    pb.close()
    with torch.no_grad():
        ppl_test = evaluate(model, testloader, device)
    print(f'Final test ppl: {ppl_test}')
    print('Saving model...')
    torch.save(
        {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'opt': opt},
        os.path.join(opt.xp_dir, 'model.pth')
    )
    print('Done')


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    # -- data
    parser.add_argument('--dataroot', type=str, default='data', help='Path to base data dir')
    parser.add_argument('--corpus', type=str, required=True, help='Name of corpus (s2|nyt)')
    parser.add_argument('--task', type=str, required=True, help='Task name (modeling|imputation|prediction)')
    parser.add_argument('--bert_cache_dir', type=str, default='bert_cache', help='Where to download bert tokenizer')
    # -- xp
    parser.add_argument('--xp_dir', type=str, required=True, help='Xp results will be saved here')
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    # -- model
    parser.add_argument('--nwe', type=int, default=400, help='size of word embedding')
    parser.add_argument('--nha', type=int, default=30, help='size of static author representation')
    parser.add_argument('--nhat', type=int, default=10, help='size of dynamic author representation')
    parser.add_argument('--cond_fusion', type=str, default='w0',
                        help='how to fuse conditioning vectors [ha, hat] into rnn language mode? (w0|h0|cat)')
    parser.add_argument('--nhid_dyn', type=int, default=64, help='size of the dynamic function hidden vectors')
    parser.add_argument('--nlayers_dyn', type=int, default=3, help='number of layers in the dynamic function')
    parser.add_argument('--nhid_lm', type=int, default=400, help='size of rnn lm state vector')
    parser.add_argument('--nlayers_lm', type=int, default=2, help='number of layers in rnn lm')
    parser.add_argument('--dropouti', type=float, default=0.5, help='input dropout')
    parser.add_argument('--dropoutl', type=float, default=0.1, help='dropout between rnn lm layers')
    parser.add_argument('--dropouto', type=float, default=0.6, help='output dropout')
    parser.add_argument('--dropoutw', type=float, default=0.4, help='weight dropout in rnn lm')
    parser.add_argument('--tie_weights', action='store_true', help='tie embeddings and decoder weights?')
    # -- regularization
    parser.add_argument('--l2_a', type=float, default=0., help='L2 regularization static author representations')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    # -- optimizer
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--lr_scheduling_burnin', type=int, default=50000, help='number of iter without lr scheduling')
    parser.add_argument('--lr_scheduling_niter', type=int, default=20000, help='number of iter with linear lr scheduling')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # -- checkpoints
    parser.add_argument('--chkpt_interval', type=int, default=10000, help='number of iter between eval')
    # -- cuda
    parser.add_argument('--device', default='-1', help='-1: cpu; > -1: cuda device id')
    # parse
    opt = parser.parse_args()
    # main
    main(opt)

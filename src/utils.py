
import os
import errno
import torch
import sys
import logging
import json
from pathlib import Path
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.distributed as dist
import csv
import os.path as osp
import time
import re
import pdb
from torch import nn
from numpy import mean
import multiprocessing
import math
import random
import numpy as np
import scipy
import scipy.sparse as sp


def set_optim(opt, model_list, freeze_part=[], accumulation_step=None):
    named_parameters = []
    param_name = []
    for model in model_list:
        model_para_train, freeze_layer = [], []
        model_para = list(model.named_parameters())

        for n, p in model_para:
            if not any(nd in n for nd in freeze_part):
                model_para_train.append((n, p))
                param_name.append(n)
            else:
                p.requires_grad = False
                freeze_layer.append((n, p))
        # pdb.set_trace()
        named_parameters.extend(model_para_train)

    parameters = [
        {'params': [p for n, p in named_parameters], "lr": opt.lr, 'weight_decay': opt.weight_decay}
    ]

    if opt.optim == 'adamw':
        # optimizer = optim.AdamW(model.parameters(), lr=opt.lr, eps=opt.adam_epsilon)
        optimizer = optim.AdamW(parameters, lr=opt.lr, eps=opt.adam_epsilon)
        # optimizer = AdamW(parameters, lr=opt.lr, eps=opt.adam_epsilon)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=opt.lr)

    if accumulation_step is None:
        accumulation_step = opt.accumulation_steps
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        scheduler_steps = opt.total_steps
        # scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps / accumulation_step), num_training_steps=int(opt.total_steps / accumulation_step))
    elif opt.scheduler == 'cos':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_steps / accumulation_step), num_training_steps=int(opt.total_steps / accumulation_step))

    return optimizer, scheduler


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.0


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        # self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        # if self.fixed_lr:
        #     return 1.0

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
                   )


class Loss_log():
    def __init__(self):
        self.loss = [999999.]
        self.acc = [0.]
        self.flag = 0
        self.token_right_num = []
        self.token_all_num = []
        self.use_top_k_acc = 0

    def acc_init(self, topn=[1]):
        self.loss = []
        self.token_right_num = []
        self.token_all_num = []
        self.topn = topn
        self.use_top_k_acc = 1
        self.top_k_word_right = {}
        for n in topn:
            self.top_k_word_right[n] = []

    def get_token_acc(self):
        if len(self.token_all_num) == 0:
            return 0.
        elif self.use_top_k_acc == 1:
            res = []
            for n in self.topn:
                res.append(round((sum(self.top_k_word_right[n]) / sum(self.token_all_num)) * 100, 3))
            return res
        else:
            return [sum(self.token_right_num) / sum(self.token_all_num)]

    def update_token(self, token_num, token_right):
        self.token_all_num.append(token_num)
        if isinstance(token_right, list):
            for i, n in enumerate(self.topn):
                self.top_k_word_right[n].append(token_right[i])
        self.token_right_num.append(token_right)

    def update(self, case):
        self.loss.append(case)

    def update_acc(self, case):
        self.acc.append(case)

    def get_acc(self):
        return self.acc[-1]

    def get_min_loss(self):
        return min(self.loss)

    def get_loss(self):
        if len(self.loss) == 0:
            return 500.
        return mean(self.loss)

    def early_stop(self):
        # min_loss = min(self.loss)
        if self.loss[-1] > min(self.loss):
            self.flag += 1
        else:
            self.flag = 0

        if self.flag > 1000:
            return True
        else:
            return False

    def torch_accuracy(output, target, topk=(1,)):
        '''
        param output, target: should be torch Variable
        '''
        # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
        # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
        # print(type(output))

        topn = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(topn, 1, True, True)
        pred = pred.t()

        is_correct = pred.eq(target.view(1, -1).expand_as(pred))

        ans = []
        ans_num = []
        for i in topk:
            # is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
            is_correct_i = is_correct[:i].contiguous().view(-1).float().sum(0, keepdim=True)
            ans_num.append(int(is_correct_i.item()))
            ans.append(is_correct_i.mul_(100.0 / batch_size))

        return ans, ans_num


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    distance = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(distance, 0.0, np.inf)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def multi_cal_neg(pos_triples, task, triples, r_hs_dict, r_ts_dict, ids, neg_scope):
    neg_triples = list()
    for idx, tas in enumerate(task):
        (h, r, t) = pos_triples[tas]
        h2, r2, t2 = h, r, t
        temp_scope, num = neg_scope, 0
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                if temp_scope:
                    h2 = random.sample(r_hs_dict[r], 1)[0]
                else:
                    for id in ids:
                        if h2 in id:
                            h2 = random.sample(id, 1)[0]
                            break
            else:
                if temp_scope:
                    t2 = random.sample(r_ts_dict[r], 1)[0]
                else:
                    for id in ids:
                        if t2 in id:
                            t2 = random.sample(id, 1)[0]
                            break
            if (h2, r2, t2) not in triples:
                break
            else:
                num += 1
                if num > 10:
                    temp_scope = False
        neg_triples.append((h2, r2, t2))
    return neg_triples


def multi_typed_sampling(pos_triples, triples, r_hs_dict, r_ts_dict, ids, neg_scope):
    t_ = time.time()
    triples = set(triples)
    tasks = div_list(np.array(range(len(pos_triples)), dtype=np.int32), 10)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(multi_cal_neg, (pos_triples, task, triples, r_hs_dict, r_ts_dict, ids, neg_scope)))
    pool.close()
    pool.join()
    neg_triples = []
    for res in reses:
        neg_triples.extend(res.get())
    return neg_triples


def nearest_neighbor_sampling(emb, left, right, K):
    t = time.time()
    neg_left = []
    distance = pairwise_distances(emb[right], emb[right])
    for idx in range(right.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_left.append(right[indices[1:K + 1]])
    neg_left = torch.cat(tuple(neg_left), dim=0)
    neg_right = []
    distance = pairwise_distances(emb[left], emb[left])
    for idx in range(left.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_right.append(left[indices[1:K + 1]])
    neg_right = torch.cat(tuple(neg_right), dim=0)
    return neg_left, neg_right


def get_adjr(ent_size, triples, norm=False):
    print('getting a sparse tensor r_adj...')
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1
    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])

    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)

    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        # 1. normalize_adj
        # 2. Convert a scipy sparse matrix to a torch sparse tensor
        # pdb.set_trace()
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size]))
        return M


def multi_cal_rank(task, sim, top_k, l_or_r):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    for i in range(len(task)):
        ref = task[i]
        if l_or_r == 0:
            rank = (sim[i, :]).argsort()
        else:
            rank = (sim[:, i]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, num, mrr


def multi_get_hits(Lvec, Rvec, top_k=(1, 5, 10, 50, 100), args=None):
    result = []
    sim = pairwise_distances(torch.FloatTensor(Lvec), torch.FloatTensor(Rvec)).numpy()
    if args.csls is True:
        sim = 1 - csls_sim(1 - sim, args.csls_k)
    for i in [0, 1]:
        top_total = np.array([0] * len(top_k))
        mean_total, mrr_total = 0.0, 0.0
        s_len = Lvec.shape[0] if i == 0 else Rvec.shape[0]
        tasks = div_list(np.array(range(s_len)), 10)
        pool = multiprocessing.Pool(processes=len(tasks))
        reses = list()
        for task in tasks:
            if i == 0:
                reses.append(pool.apply_async(multi_cal_rank, (task, sim[task, :], top_k, i)))
            else:
                reses.append(pool.apply_async(multi_cal_rank, (task, sim[:, task], top_k, i)))
        pool.close()
        pool.join()
        for res in reses:
            mean, num, mrr = res.get()
            mean_total += mean
            mrr_total += mrr
            top_total += np.array(num)
        acc_total = top_total / s_len
        for i in range(len(acc_total)):
            acc_total[i] = round(acc_total[i], 4)
        mean_total /= s_len
        mrr_total /= s_len
        result.append(acc_total)
        result.append(mean_total)
        result.append(mrr_total)
    return result


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.
    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.
    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = torch.mean(torch.topk(sim_mat, k)[0], 1)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(), k)[0], 1)
    csls_sim_mat = 2 * sim_mat.t() - nearest_values1
    csls_sim_mat = csls_sim_mat.t() - nearest_values2
    return csls_sim_mat


def get_topk_indices(M, K=1000):
    H, W = M.shape
    M_view = M.view(-1)
    vals, indices = M_view.topk(K)
    print("highest sim:", vals[0].item(), "lowest sim:", vals[-1].item())
    two_d_indices = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
    return two_d_indices


def normalize_zero_one(A):
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0]
    return A


def output_device(model):
    sd = model.state_dict()
    devices = []
    for v in sd.values():
        if v.device not in devices:
            devices.append(v.device)
    # for d in devices:
    #     print(d)
    print(devices)

import torch
from torch import nn
import torch.nn.functional as F
import pdb
import numpy as np


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class icl_loss(nn.Module):

    def __init__(self, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0, inversion=False, replay=False, neg_cross_kg=False):
        super(icl_loss, self).__init__()
        self.tau = tau
        self.sim = cosine_sim
        self.weight = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view
        self.intra_weight = intra_weight  # the factor of aa and bb
        self.inversion = inversion
        self.replay = replay
        self.neg_cross_kg = neg_cross_kg

    def softXEnt(self, target, logits, replay=False, neg_cross_kg=False):
        # torch.Size([2239, 4478])

        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        # .unsqueeze(1)
        if replay:
            logits = logits
            idx = torch.arange(start=0, end=logprobs.shape[0], dtype=torch.int64).cuda()
            # 如果希望限定负样本必须在对面的KG可以在这里加限定
            # if neg_cross_kg:
            stg_neg = logits.argmax(dim=1)
            new_value = torch.zeros(logprobs.shape[0]).cuda()
            index = (
                idx,
                stg_neg,
            )
            # 目的是为了求第二大的值
            logits = logits.index_put(index, new_value)
            # pdb.set_trace()
            stg_neg_2 = logits.argmax(dim=1)
            # idx = torch.cat([idx, stg_neg], dim=1)
            # 判断哪些地方相等
            tmp = idx.eq_(stg_neg)
            # 相等（GT）的地方需要替换为第二大的
            neg_idx = stg_neg - stg_neg * tmp + stg_neg_2 * tmp
            return loss, neg_idx

        return loss

    # replay 相当于是扩充当前的train_link
    # 同一个复杂样本不应该超过两次：
    # 格式：
    # train_links[:, 0]: shape: (2239,)
    # array([11303,  2910,  2072, ..., 10504, 13555,  8416], dtype=int32)
    def forward(self, emb, train_links, neg_l=None, neg_r=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
        num_ent = emb.shape[0]
        # Get (normalized) hidden1 and hidden2.
        zis = emb[train_links[:, 0]]
        zjs = emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight
        # 2
        n_view = self.n_view
        LARGE_NUM = 1e9
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        hidden1_large = hidden1
        hidden2_large = hidden2

        if neg_l is None:
            num_classes = batch_size * n_view
        else:
            num_classes = batch_size * n_view + neg_l.shape[0]
            num_classes_2 = batch_size * n_view + neg_r.shape[0]

        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.cuda()
        if neg_l is not None:
            labels_2 = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes_2).float()
            labels_2 = labels_2.cuda()

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.cuda().float()
        # 自己和自己不要算相似度
        # 其他的相似度要尽可能小
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM

        # 自己和自己不要算相似度
        # 其他的相似度要尽可能小
        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        # TODO: 自己和自己的负样本算相似度, 要尽可能小, 不需要减mask
        if neg_l is not None:
            zins = emb[neg_l]
            zjns = emb[neg_r]
            logits_ana = torch.matmul(hidden1, torch.transpose(zins, 0, 1)) / temperature
            logits_bnb = torch.matmul(hidden2, torch.transpose(zjns, 0, 1)) / temperature

        # 自己和匹配的需要算相似度，并且用标签约束
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
        # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
        if self.inversion:
            logits_a = torch.cat([logits_ab, logits_bb], dim=1)
            logits_b = torch.cat([logits_ba, logits_aa], dim=1)
        else:
            if neg_l is None:
                logits_a = torch.cat([logits_ab, logits_aa], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb], dim=1)
            else:
                logits_a = torch.cat([logits_ab, logits_aa, logits_ana], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb, logits_bnb], dim=1)

        # 除了shape，所以不需要扩bs能保证loss的稳定

        if self.replay:
            loss_a, a_neg_idx = self.softXEnt(labels, logits_a, replay=True, neg_cross_kg=self.neg_cross_kg)
            if neg_l is not None:
                loss_b, b_neg_idx = self.softXEnt(labels_2, logits_b, replay=True, neg_cross_kg=self.neg_cross_kg)
                #
                a_ea_cand = torch.cat([train_links[:, 1], train_links[:, 0], neg_l]).cuda()
                b_ea_cand = torch.cat([train_links[:, 0], train_links[:, 1], neg_r]).cuda()
            else:
                # 刚开始的时候是空的可能
                loss_b, b_neg_idx = self.softXEnt(labels, logits_b, replay=True, neg_cross_kg=self.neg_cross_kg)
                a_ea_cand = torch.cat([train_links[:, 1], train_links[:, 0]]).cuda()
                b_ea_cand = torch.cat([train_links[:, 0], train_links[:, 1]]).cuda()

            a_neg = a_ea_cand[a_neg_idx]
            b_neg = b_ea_cand[b_neg_idx]
            # 左边的负样本，右边的负样本
            return alpha * loss_a + (1 - alpha) * loss_b, a_neg, b_neg

        else:
            loss_a = self.softXEnt(labels, logits_a)
            loss_b = self.softXEnt(labels, logits_b)
            return alpha * loss_a + (1 - alpha) * loss_b


class ial_loss(nn.Module):
    """
    unimodal-multimodal kl loss
    """

    def __init__(self, tau=0.05, ab_weight=0.5, zoom=0.1,
                 n_view=2, inversion=False,
                 reduction="mean", detach=False):
        super(ial_loss, self).__init__()
        self.tau = tau
        self.sim = cosine_sim
        self.weight = ab_weight
        self.zoom = zoom
        self.n_view = n_view
        self.inversion = inversion
        self.reduction = reduction
        self.detach = detach

    def forward(self, src_emb, tar_emb, train_links, norm=True):
        if norm:
            src_emb = F.normalize(src_emb, dim=1)
            tar_emb = F.normalize(tar_emb, dim=1)

        # Get (normalized) hidden1 and hidden2.
        src_zis = src_emb[train_links[:, 0]]
        src_zjs = src_emb[train_links[:, 1]]
        tar_zis = tar_emb[train_links[:, 0]]
        tar_zjs = tar_emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight

        assert src_zis.shape[0] == tar_zjs.shape[0]
        batch_size = src_zis.shape[0]
        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.cuda().float()
        p_ab = torch.matmul(src_zis, torch.transpose(src_zjs, 0, 1)) / temperature
        p_ba = torch.matmul(src_zjs, torch.transpose(src_zis, 0, 1)) / temperature
        q_ab = torch.matmul(tar_zis, torch.transpose(tar_zjs, 0, 1)) / temperature
        q_ba = torch.matmul(tar_zjs, torch.transpose(tar_zis, 0, 1)) / temperature
        # add self-contrastive
        p_aa = torch.matmul(src_zis, torch.transpose(src_zis, 0, 1)) / temperature
        p_bb = torch.matmul(src_zjs, torch.transpose(src_zjs, 0, 1)) / temperature
        q_aa = torch.matmul(tar_zis, torch.transpose(tar_zis, 0, 1)) / temperature
        q_bb = torch.matmul(tar_zjs, torch.transpose(tar_zjs, 0, 1)) / temperature
        p_aa = p_aa - masks * LARGE_NUM
        p_bb = p_bb - masks * LARGE_NUM
        q_aa = q_aa - masks * LARGE_NUM
        q_bb = q_bb - masks * LARGE_NUM

        if self.inversion:
            p_ab = torch.cat([p_ab, p_bb], dim=1)
            p_ba = torch.cat([p_ba, p_aa], dim=1)
            q_ab = torch.cat([q_ab, q_bb], dim=1)
            q_ba = torch.cat([q_ba, q_aa], dim=1)
        else:
            p_ab = torch.cat([p_ab, p_aa], dim=1)
            p_ba = torch.cat([p_ba, p_bb], dim=1)
            q_ab = torch.cat([q_ab, q_aa], dim=1)
            q_ba = torch.cat([q_ba, q_bb], dim=1)

        # param 1 need to log_softmax, param 2 need to softmax
        loss_a = F.kl_div(F.log_softmax(p_ab, dim=1), F.softmax(q_ab.detach(), dim=1), reduction="none")
        loss_b = F.kl_div(F.log_softmax(p_ba, dim=1), F.softmax(q_ba.detach(), dim=1), reduction="none")

        if self.reduction == "mean":
            loss_a = loss_a.mean()
            loss_b = loss_b.mean()
        elif self.reduction == "sum":
            loss_a = loss_a.sum()
            loss_b = loss_b.sum()
        # The purpose of the zoom is to narrow the range of losses
        return self.zoom * (alpha * loss_a + (1 - alpha) * loss_b)

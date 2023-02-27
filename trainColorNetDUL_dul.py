import torch
from torch import nn
import json

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch.utils.data import DataLoader, Dataset
import cv2
import os
from os.path import join
import random
import torch.optim as optimzer
import numpy as np
import glob
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import ImageFilter, ImageOps, Image
from torchvision import datasets, transforms
import time
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets import dataloader_video
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from models.resnet_my import resnet18
import matplotlib.pyplot as plt
import argparse

width = 256
height = 256
stride = 8
patch_dim = stride * stride
TEMP = 25


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

def l1_norm(input, axis=1):
    norm = torch.norm(input, 1, axis, True)
    norm = torch.clip(norm, 0.00001)
    output = torch.div(input, norm)
    return output


class ViT(nn.Module):
    def __init__(self, cfg, ohem_range=[0.4, 1.0]):
        super().__init__()

        self.cls_num = 16
        self.cfg = cfg
        self.ohem_range = ohem_range
        print('net ohem range', ohem_range)

        self.convlocal = resnet18(is_thin=True, first_kernal_size = 3)
        self.eye = None

    def _align(self, x, t):
        tf = F.affine_grid(t, size=x.size(), align_corners=False)
        return F.grid_sample(x, tf, align_corners=False, mode="nearest")

    def _key_val(self, ctr, q):
        """
        Args:
            ctr: [N,K]
            q: [BHW,K]
        Returns:
            val: [BHW,N]
        """

        # [BHW,K] x [N,K].t -> [BHWxN]
        vals = torch.mm(q, ctr.t())  # [BHW,N]

        # normalising attention
        return vals / self.cfg.TEST.TEMP

    def fetch_first(self, x1, x2, T):
        assert x1.shape[1:] == x2.shape[1:]
        c, h, w = x1.shape[1:]

        x1 = x1.view(-1, T + 1, c, h, w)
        x2 = x2.view(-1, T - 1, c, h, w)

        x2 = torch.cat((x1[:, -1:], x2), 1)
        x1 = x1[:, :-1]

        return x1.flatten(0, 1), x2.flatten(0, 1)

    def _sample_index(self, x, T, N):
        """Sample indices of the anchors

        Args:
            x: [BT,K,H,W]
        Returns:
            index: [B,N*N,K]
        """

        BT, K, H, W = x.shape
        B = x.view(-1, T, K, H * W).shape[0]

        # sample indices from a uniform grid
        xs, ys = W // N, H // N
        x_sample = torch.arange(0, W, xs).view(1, 1, N)
        y_sample = torch.arange(0, H, ys).view(1, N, 1)

        # Random offsets
        # [B x 1 x N]
        x_sample = x_sample + torch.randint(0, xs, (B, 1, 1))
        # [B x N x 1]
        y_sample = y_sample + torch.randint(0, ys, (B, 1, 1))

        # batch index
        # [B x N x N]
        hw_index = torch.LongTensor(x_sample + y_sample * W)

        return hw_index

    def _sample_from(self, x, index, T, N):
        """Gather the features based on the index

        Args:
            x: [BT,K,H,W]
            index: [B,N,N] defines the indices of NxN grid for a single
                           frame in each of B video clips
        Returns:
            anchors: [BNN,K] sampled features given by index from x
        """

        BT, K, H, W = x.shape
        x = x.view(-1, T, K, H * W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0, 1, 3, 2]).reshape(B, -1, K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B, -1, 1).expand(-1, -1, K)

        # selecting from the uniform grid
        y = x.gather(1, index.to(x.device))

        # [BNN,K]
        return y.flatten(0, 1)

    def _pseudo_mask(self, logits, T):
        BT, K, h, w = logits.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.cfg.MODEL.GRID_SIZE ** 2

        # generating a pseudo label
        # first we need to mask out the affinities across the batch
        if self.eye is None or self.eye.shape[0] != B * T \
                or self.eye.shape[1] != B * N:
            eye = torch.eye(B)[:, :, None].expand(-1, -1, N).reshape(B, -1)
            eye = eye.unsqueeze(1).expand(-1, T, -1).reshape(B * T, B * N, 1, 1)
            self.eye = eye.to(logits.device)

        probs = F.softmax(logits, 1)
        return probs * self.eye

    def _cluster_grid(self, k1, k2, aff1, aff2, T, index=None):
        """ Selecting clusters within a sequence
        Args:
            k1: [BT,K,H,W]
            k2: [BT,K,H,W]
        """

        BT, K, H, W = k1.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.cfg.MODEL.GRID_SIZE ** 2

        # [BT,K,H,W] -> [BTHW,K]
        flatten = lambda x: x.flatten(2, 3).permute([0, 2, 1]).flatten(0, 1)

        # [BTHW,BN] -> [BT,BN,H,W]
        def unflatten(x, aff=None):
            x = x.view(BT, H * W, -1).permute([0, 2, 1]).view(BT, -1, H, W)
            if aff is None:
                return x
            return self._align(x, aff)

        index = self._sample_index(k1, T, N=self.cfg.MODEL.GRID_SIZE)
        query1 = self._sample_from(k1, index, T, N=self.cfg.MODEL.GRID_SIZE)

        """Computing the distances and pseudo labels"""

        # [BTHW,K]
        k1_ = flatten(k1)
        k2_ = flatten(k2)

        # [BTHW,BN] -> [BTHW,BN] -> [BT,BN,H,W]
        vals_soft = unflatten(self._key_val(query1, k1_), aff1)
        vals_pseudo = unflatten(self._key_val(query1, k2_), aff2)

        # [BT,BN,H,W]
        probs_pseudo = self._pseudo_mask(vals_pseudo, T)
        probs_pseudo2 = self._pseudo_mask(vals_soft, T)

        pseudo = probs_pseudo.argmax(1)
        pseudo2 = probs_pseudo2.argmax(1)

        return vals_soft, pseudo, index

    def _ce_loss2(self, k1, k2, aff1, T, H, W):
        """ Selecting clusters within a sequence
        Args:
            k1: [B,T,HW,K]
            k2: [B,T,HW,K]
            aff1: [B,T,2,3]
        """
        B, _, _, K = k1.size()

        # loss_ce = F.cross_entropy(vals_soft, pseudo)
        k1 = k1.reshape(B * T, H, W, K).permute(0, 3, 1, 2)
        k2 = k2.reshape(B * T, H, W, K).permute(0, 3, 1, 2)

        loss_ce = F.smooth_l1_loss(self._align_my(k1, aff1), k2, beta=0.5, reduction='none')
        loss_ce = torch.mean(loss_ce, [1, 2, 3])
        loss_ce = loss_ce.sort()[0]
        idx1 = int(loss_ce.size(0) * self.ohem_range[0])
        idx2 = int(loss_ce.size(0) * self.ohem_range[1])
        loss_ce = torch.mean(loss_ce[idx1:idx2])
        return loss_ce

    def _ref_loss(self, x, y, N=4):
        B, _, h, w = x.shape

        index = self._sample_index(x, T=1, N=N)
        x1 = self._sample_from(x, index, T=1, N=N)
        y1 = self._sample_from(y, index, T=1, N=N)
        logits = torch.mm(x1, y1.t()) / self.cfg.TEST.TEMP

        labels = torch.arange(logits.size(1)).to(logits.device)
        loss_ref = F.cross_entropy(logits, labels, reduction='none')

        loss_ref = loss_ref.sort()[0]
        idx1 = int(loss_ref.size(0) * self.ohem_range[0])
        idx2 = int(loss_ref.size(0) * self.ohem_range[1])
        loss_ref = torch.mean(loss_ref[idx1:idx2])
        return loss_ref

    def _ce_loss(self, x, pseudo_map, T, eps=1e-5):
        error_map = F.cross_entropy(x, pseudo_map, reduction="none", ignore_index=-1)

        BT, h, w = error_map.shape
        errors = error_map.view(-1, T, h, w)
        error_ref, error_t = errors[:, 0], errors[:, 1:]
        error_t = error_t.reshape(-1)
        error_t = error_t.sort()[0]
        idx1 = int(error_t.size(0) * self.ohem_range[0])
        idx2 = int(error_t.size(0) * self.ohem_range[1])
        error_t = torch.mean(error_t[idx1:idx2])
        return error_t

    def _neg_loss(self, feat):
        '''
        feat:[B,HW,C]
        '''
        B, HW, C = feat.size()
        neg_loss = 0
        for i in range(B):
            for j in range(B):
                if i == j:
                    continue
                neg_loss += torch.mean(torch.abs(torch.mm(feat[i], feat[j].permute(1, 0))))
        return neg_loss / B / (B - 1)

    # def _ce_loss(self, x, pseudo_map, T, eps=1e-5):
    #     error_map = F.cross_entropy(x, pseudo_map, reduction="none", ignore_index=-1)
    #
    #     BT,h,w = error_map.shape
    #     errors = error_map.view(-1,T,h,w)
    #     error_ref, error_t = errors[:,0], errors[:,1:]
    #
    #     return error_ref.mean(), error_t.mean(), error_map

    def forward(self, frames1_d , frames2_d ,affines_d , affine_src_d):
        result = {}
        # ==========================DUL================================
        B, T, _, _, _ = frames1_d.size()
        images1_d = torch.cat((frames1_d, frames2_d[:, ::T]), 1)
        images1_d = images1_d.flatten(0, 1).cuda()
        images2_d = frames2_d[:, 1:].flatten(0, 1).cuda()

        affines_d = affines_d.flatten(0, 1).cuda()
        affine_src_d = affine_src_d.flatten(0, 1).cuda()


        _, key1_cls, _, _ = self.convlocal(images1_d)
        _, C, H, W = key1_cls.size()

        with torch.no_grad():
            _, key2_cls, _, _ = self.convlocal(images2_d)

        key1_cls, key2_cls = self.fetch_first(key1_cls, key2_cls, T)

        vals, pseudo, index = self._cluster_grid(key1_cls, key2_cls, affines_d, affine_src_d, T)

        key1_aligned = self._align(key1_cls, affines_d)
        key2_aligned = self._align(key2_cls, affine_src_d)

        n_ref = self.cfg.MODEL.GRID_SIZE_REF  # N = 4
        loss_cross_key = self._ref_loss(key1_aligned[::T], key2_aligned[::T], N=n_ref)
        loss_temp = self._ce_loss(vals, pseudo, T)
        loss_dul = 0.1 * loss_cross_key + loss_temp
        result['loss_dul'] = loss_dul

        # =============================================================
        return result


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in [100, 200, 300, 400]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
            print('changed lr:{}'.format(param_group['lr']))


def adjust_learning_rate_cos(optimizer, epoch):
    epoch = min(260, epoch)
    epochs = 300
    warmup_epochs = 25
    lr = 0.0001
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = lr * 0.5 * (1. + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('changed lr:{}'.format(lr))
    return lr


def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    init_random_seed(777)
    cfg_from_file(r'./configs/ytvos.yaml')
    cfg.MODEL.FEATURE_DIM = 128
    cfg.DATASET.VIDEO_LEN = 5
    cfg.DATASET.RND_ZOOM_RANGE = [0.5, 1.]

    parser = argparse.ArgumentParser('')
    parser.add_argument('--exp', default='benchmark', type=str, help="实验信息")
    parser.add_argument('--is_ohem', action='store_true')
    args = parser.parse_args()

    expid = args.exp
    if args.is_ohem:
        ohem_range = [0.35,0.95]
    else:
        ohem_range = [0.0, 1.0]


    net = ViT(cfg, ohem_range=ohem_range).cuda()
    net.cuda()
    print(net.convlocal.conv1)
    print(net.convlocal.layer6)

    optimzer = optimzer.AdamW(net.parameters(), lr=0.0001, eps=1e-8, betas=[0.9, 0.95])
    dataset = dataloader_video.DataVideo(cfg, 'train_ytvos')
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=12, drop_last=True)
    color_lst = torch.randint(0, 255, (net.cls_num, 3)).cuda()

    for epoch in range(1, 301):
        adjust_learning_rate_cos(optimzer, epoch)
        for i, batch in enumerate(dataloader):

            frames1_d, frames2_d, affine1_d, affine_src_d = batch
            frames1_d = frames1_d.cuda().float()
            frames2_d = frames2_d.cuda().float()
            affine1_d = affine1_d.cuda().float()
            affine_src_d = affine_src_d.cuda().float()
            optimzer.zero_grad()

            pred_result = net(frames1_d,frames2_d, affine1_d, affine_src_d)

            loss = pred_result['loss_dul']
            loss.backward()
            optimzer.step()

            timeStr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print(expid,'{:0>3d}-{:0>6d} {} {:.5f}'.format(epoch, i, timeStr, loss.item() ))

        if epoch % 20 == 0:
            net_sd = net.convlocal.state_dict()
            save_dict = {
                'model': net_sd,
                'train_config': {'expid':expid,
                                 'ohem_range':ohem_range,
                                 'is_thin':True,
                                 'first_kernal_size':3
                                 }
            }
            torch.save(save_dict,
                           './snapshots/{}_{:0>3d}.pkl'.format(expid,epoch))



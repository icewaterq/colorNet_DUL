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
from datasets import dataloader_video_my_coco, dataloader_video_my_dul
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from models.resnet_my import resnet18
import matplotlib.pyplot as plt
import argparse

width = 256
height = 256
stride = 8
patch_dim = stride * stride

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

def l1_norm(input, axis=1):
    norm = torch.norm(input, 1, axis, True)
    norm = torch.clip(norm, 0.00001)
    output = torch.div(input, norm)
    return output


class ViT(nn.Module):
    def __init__(self, cfg, ohem_range=[0.4, 1.0], model_size = 'L', is_dul = False, is_neg = True, first_kernal_size = 3, TEMP = 25):
        super().__init__()

        self.cls_num = 16
        self.cfg = cfg
        self.ohem_range = ohem_range
        print('net ohem range', ohem_range)

        self.eye = None
        self.is_dul = is_dul
        self.is_neg = is_neg
        self.TEMP = TEMP
        feat_mode = 'color_dul'
        if not is_dul:
            feat_mode = 'color'

        print('.....init network.')
        print('is_dul', self.is_dul)
        print('is_neg', self.is_neg)
        print('TEMP', self.TEMP)
        print('feat_mode',feat_mode)

        self.convlocal = resnet18(model_size = model_size, first_kernal_size = first_kernal_size, feat_mode = feat_mode)

    def _align_my(self, x, tf):
        return F.grid_sample(x, tf, align_corners=False, mode="nearest", padding_mode='reflection')

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
        return F.cross_entropy(logits, labels)

    def _ce_loss(self, x, pseudo_map, T, eps=1e-5):
        error_map = F.cross_entropy(x, pseudo_map, reduction="none", ignore_index=-1)

        BT, h, w = error_map.shape
        errors = error_map.view(-1, T, h, w)
        error_ref, error_t = errors[:, 0], errors[:, 1:]

        return error_t.mean()

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

    def forward(self, frames1, frames2, affines, affine_src, labLst, w_mask_lst, frames1_d = None, frames2_d = None,
                affines_d = None, affine_src_d = None):
        result = {}
        result['feat1'] = None
        result['feat2'] = None
        result['loss_ce'] = frames1.new_zeros(1)
        result['neg_loss'] = frames1.new_zeros(1)
        result['color_lst'] = None
        result['loss_color'] = frames1.new_zeros(1)
        result['loss_dul'] = frames1.new_zeros(1)

        B, T, _, _, _ = frames1.size()
        img1 = frames1.flatten(0, 1)
        img2 = frames2.flatten(0, 1)
        affines = affines.flatten(0, 1).cuda()
        affine_src = affine_src.flatten(0, 1).cuda()

        # ==========================DUL================================
        if self.is_dul \
                and not frames1_d is None \
                and not frames2_d is None \
                and not affines_d is None \
                and not affine_src_d is None:
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
        key1, _, _, _ = self.convlocal(img1)
        _, C, H, W = key1.size()

        with torch.no_grad():
            key2, _, _, _ = self.convlocal(img2)

        key2 = key2.permute(0, 2, 3, 1).reshape(B, T, -1, C)  # B,T,HW,C
        key1 = key1.permute(0, 2, 3, 1).reshape(B, T, -1, C)
        #Lab图像
        labLst = labLst.view(-1, 3, int(height // stride), int(stride),
                             int(width // stride), int(stride)).permute(0, 2, 4, 3, 5, 1).reshape(B, T, H * W,
                                                                                                 patch_dim * 3)
        # 边缘权重图
        w_mask_lst = w_mask_lst.view(-1, 3, int(height // stride), int(stride),
                                     int(width // stride), int(stride)).permute(0, 2, 4, 3, 5, 1).reshape(B, T, H * W,
                                                                                                          patch_dim * 3)
        #空间一致性Loss
        loss_ce = self._ce_loss2(key1, key2, affines, T, H, W)

        color_loss_lst = []
        color_pred_lst = []

        HW = H * W
        if self.is_neg:
            for i in range(self.cfg.DATASET.VIDEO_LEN):
                #翻转batch顺序，拼接负样本
                featp1 = torch.cat([key1[:, i], torch.flip(key1[:, i], [0])], 1)  # key : B,2HW,C
                lab1 = torch.cat([labLst[:, i], torch.flip(labLst[:, i], [0])], 1)  # key : B,2HW,C
                for j in range(self.cfg.DATASET.VIDEO_LEN):
                    if i == j:
                        continue
                    featp2 = torch.cat([key1[:, j], torch.flip(key1[:, j], [0])], 1)
                    att1 = torch.bmm(featp2[:, :HW], featp1.permute(0, 2, 1)) *  self.TEMP
                    att1 = F.softmax(att1, dim=2)  # [B,HW,HW2]

                    gt1 = labLst[:, j]
                    mask = w_mask_lst[:, j]
                    color1 = torch.bmm(att1, lab1)
                    for bid in range(B):
                        loss_color_part = F.smooth_l1_loss(color1[bid] * mask[bid], gt1[bid] * mask[bid], beta=0.5)
                        color_loss_lst.append(loss_color_part)


                    if i == 0:
                        color1 = color1.view(-1, int(height // stride), int(width // stride),
                                             stride, stride, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, height, width)
                        color_pred_lst.append(color1)

        else:
            for i in range(self.cfg.DATASET.VIDEO_LEN):
                featp1 = key1[:, i]  # key : B,HW,C
                lab1 = labLst[:, i]  # key : B,HW,C
                for j in range(self.cfg.DATASET.VIDEO_LEN):
                    if i == j:
                        continue
                    featp2 = key1[:, j]
                    att1 = torch.bmm(featp2, featp1.permute(0, 2, 1)) *  self.TEMP
                    att1 = F.softmax(att1, dim=2)  # [B,HW,HW]

                    gt1 = labLst[:, j]
                    mask = w_mask_lst[:, j]
                    color1 = torch.bmm(att1, lab1)
                    for bid in range(B):
                        loss_color_part = F.smooth_l1_loss(color1[bid] * mask[bid], gt1[bid] * mask[bid], beta=0.5)
                        color_loss_lst.append(loss_color_part)
                    #图像重构结果
                    if i == 0:
                        color1 = color1.view(-1, int(height // stride), int(width // stride),
                                             stride, stride, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, height, width)
                        color_pred_lst.append(color1)

        #loss排序，只有指定区间内的loss进行计算并回传梯度，实际使用时用的[0,1.0]计算所有loss
        color_loss_lst.sort()
        loss_color = 0
        idx1 = int(len(color_loss_lst) * self.ohem_range[0])
        idx2 = int(len(color_loss_lst) * self.ohem_range[1])
        for i in range(idx1, idx2):
            loss_color += color_loss_lst[i]
        loss_color /= (idx2 - idx1)
        #看batch之间的相似度，实际没有使用，仅观察是否下降
        neg_loss = self._neg_loss(key1[:, 0])

        result['feat1'] = key1
        result['feat2'] = key2
        result['loss_ce'] = loss_ce
        result['neg_loss'] = neg_loss

        result['color_lst'] = color_pred_lst
        result['loss_color'] = loss_color

        return result


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in [100, 200, 300, 400]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
            print('changed lr:{}'.format(param_group['lr']))

#学习率warmup和余弦衰减
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

def load_trai_config(path):
    train_config = {
        "exp":"benchmark",
        "first_kernal_size":3,
        "color_mode":"LAB",
        "is_dul":True,
        "is_aug":True,
        "is_shadow":True,
        "is_liquid":True,
        "is_edge":True,
        "is_neg":True,
        "space_consistency": True,
        "is_labclip": True,
        "ohem_range":[0,1.0],
        "model_size":"M",
        "TEMP":25,
        "is_cos_lr":True,
        "video_len": 5
    }
    if not os.path.exists(path):
        print('train config file {} not existed, exited.'.format(path))
        exit()
    with open(path) as fp:
        jsonobj = json.load(fp)
    for key in train_config:
        if key in jsonobj:
            train_config[key] = jsonobj[key]
    return train_config


if __name__ == '__main__':
    init_random_seed(777)

    parser = argparse.ArgumentParser('')
    parser.add_argument('--exp', default='benchmark', type=str, help="实验信息")
    parser.add_argument('--train_config', default='./configs/train_config_default.json', type=str, help="训练配置文件")
    parser.add_argument('--coco', action='store_true', help="是否使用coco数据集训练")
    args = parser.parse_args()

    expid = args.exp
    train_config_path = args.train_config
    train_config = load_trai_config(train_config_path)

    print('run exp',expid)
    if args.coco:
        print('use coco train model.')
    else:
        print('use youtube-vos train model.')
    print('load train config:',train_config_path)
    for key in train_config:
        print('{} : {}'.format(key,train_config[key]))

    exp = train_config['exp']
    first_kernal_size = train_config['first_kernal_size']
    color_mode = train_config['color_mode']
    is_dul = train_config['is_dul']
    is_aug = train_config['is_aug']
    is_shadow = train_config['is_shadow']
    is_liquid = train_config['is_liquid']
    is_edge = train_config['is_edge']
    is_neg = train_config['is_neg']
    space_consistency = train_config['space_consistency']
    is_labclip = train_config['is_labclip']
    ohem_range = train_config['ohem_range']
    model_size = train_config['model_size']
    TEMP = train_config['TEMP']
    is_cos_lr = train_config['is_cos_lr']
    video_len = train_config['video_len']
    dul_loss_weight = 0.01

    cfg_from_file(r'./configs/ytvos.yaml')
    cfg.MODEL.FEATURE_DIM = 128
    cfg.DATASET.VIDEO_LEN = video_len
    cfg.DATASET.RND_ZOOM_RANGE = [0.5, 1.]

    # dataset = dataloader_video_my_dul.DataVideo(cfg, 'train_ytvos')
    # dataloader = data.DataLoader(dataset, batch_size=2, \
    #                              shuffle=True)
    #
    # for i, batch in enumerate(dataloader):
    #     # frames1, frames2, affine1,_,labs1, labs2 = batch
    #     frames1, frames2, affine1, affine_src, labs1, labs2, rgbs1, rgbs2, w_mask_lst, frames1_d, frames2_d, affine1_d, affine_src_d = batch
    #     for tt in batch:
    #         print(tt.size())
    #     assert frames1.size() == frames2.size(), "Frames shape mismatch"
    #     frames1, frames2, affine1 = frames1_d, frames2_d, affine1_d
    #     # We could simply do
    #     #   images1 = frames1.flatten(0,1).cuda()
    #     #   images2 = frames2.flatten(0,1).cuda()
    #     # Instead we pull the reference frame from the 2nd view
    #     # to the first view so that the regularising branch is
    #     # always in evaluation mode to save the GPU memory
    #     print(frames1.size(), affine1.size())
    #     # print('affine_src',affine_src[0])
    #     # print('affine_src_d',affine_src_d[0])
    #     for i in range(cfg.DATASET.VIDEO_LEN):
    #         print('affine1[0]', affine1.size())
    #         # img3 = F.grid_sample(frames1[0], affine1[0], align_corners=False, mode="bilinear",padding_mode='reflection')
    #
    #         tf = F.affine_grid(affine1[0], size=frames1[0].size(), align_corners=False)
    #         img3 = F.grid_sample(frames1[0], tf, align_corners=False, mode="bilinear")
    #
    #         img1 = frames1[0, i, :].permute(1, 2, 0).numpy()
    #         img2 = frames2[0, i, :].permute(1, 2, 0).numpy()
    #         img3 = img3[i, :].permute(1, 2, 0).numpy()
    #         mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    #         std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    #
    #         img1 = np.array((img1 * std_ + mean_) * 255, dtype=np.uint8)
    #         img2 = np.array((img2 * std_ + mean_) * 255, dtype=np.uint8)
    #         img3 = np.array((img3 * std_ + mean_) * 255, dtype=np.uint8)
    #         # img3 = cv2.resize(img3, (0, 0), fx=8, fy=8)
    #         img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    #         img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    #         img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
    #         # img1[..., 0] = 255
    #         # img2[..., 0] = 255
    #         # img2[..., 1] = img2[..., 0]
    #         # img2[..., 2] = img2[..., 0]
    #         # img3[..., 0] = 255
    #         print(img1.shape)
    #         cv2.imshow('img1', img1)
    #         cv2.imshow('img2', img2)
    #         cv2.imshow('img3', img3)
    #         plt.show()
    #         cv2.waitKey()

    net = ViT(cfg, ohem_range=ohem_range, model_size=model_size,is_dul=is_dul,is_neg = is_neg, first_kernal_size = first_kernal_size, TEMP = TEMP).cuda()
    net.cuda()
    print(net.convlocal.conv1)
    print(net.convlocal.layer6)

    optimzer = optimzer.AdamW(net.parameters(), lr=0.0001, eps=1e-8, betas=[0.9, 0.95])
    if args.coco:
        dataset = dataloader_video_my_coco.DataVideo(cfg, 'train_ytvos', is_aug=is_aug, is_shadow=is_shadow,is_dul=is_dul, is_liquid=is_liquid)
    else:
        dataset = dataloader_video_my_dul.DataVideo(cfg, 'train_ytvos',is_aug=is_aug,is_shadow=is_shadow,is_dul=is_dul,is_liquid=is_liquid)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=12, drop_last=True)
    color_lst = torch.randint(0, 255, (net.cls_num, 3)).cuda()
    for epoch in range(1, 301):
        if is_cos_lr:
            adjust_learning_rate_cos(optimzer, epoch)
        else:
            adjust_learning_rate(optimzer, epoch)
        grad_info_lst = []
        for i, batch in enumerate(dataloader):
            if is_dul:
                frames1, frames2, affine1, affine_src, labs1, labs2, rgbs1, rgbs2, w_mask_lst, frames1_d, frames2_d, affine1_d, affine_src_d = batch
                frames1_d = frames1_d.cuda().float()
                frames2_d = frames2_d.cuda().float()
                affine1_d = affine1_d.cuda().float()
                affine_src_d = affine_src_d.cuda().float()
            else:
                frames1, frames2, affine1, affine_src, labs1, labs2, rgbs1, rgbs2, w_mask_lst = batch

            frames1 = frames1.cuda().float()
            frames2 = frames2.cuda().float()
            affine1 = affine1.cuda().float()
            w_mask_lst = w_mask_lst.cuda().float()
            if color_mode == 'LAB':
                labs1 = labs1.cuda().float()
                labs2 = labs2.cuda().float()
            elif color_mode == 'RGB':
                labs1 = rgbs1.cuda().float()
                labs2 = rgbs2.cuda().float()
            elif color_mode == 'AB':
                labs1 = labs1.cuda().float()
                labs2 = labs2.cuda().float()
                labs1[:,:,0] = 0
                labs2[:,:,0] = 0
            else:
                print('illegal color mode {},only support LAB,RGB and AB.'.format(color_mode))
                exit()

            if not is_edge:
                w_mask_lst = w_mask_lst * 0 + 1
            optimzer.zero_grad()

            if is_dul:
                pred_result = net(frames1, frames2, affine1, affine_src, labs1, w_mask_lst,
                                  frames1_d,frames2_d, affine1_d, affine_src_d)
            else:
                pred_result = net(frames1, frames2, affine1, affine_src, labs1, w_mask_lst)

            loss_dul = pred_result['loss_dul']
            loss_ce = pred_result['loss_ce']
            loss_color = pred_result['loss_color']
            color_lst = pred_result['color_lst']
            neg_loss = pred_result['neg_loss']
            #由于直接将L通道置0，所以计算平均loss会比原先小，乘1.5平衡
            if color_mode == 'AB':
                loss_color *= 1.5
            if space_consistency:
                ce_weight = 1
            else:
                ce_weight = 0
            if is_dul:
                loss = loss_ce * ce_weight + loss_color + loss_dul * dul_loss_weight
            else:
                loss = loss_ce * ce_weight + loss_color

            loss.backward()
            grad_color = torch.mean(torch.abs(net.convlocal.layer4[-1].conv2.weight.grad[:net.convlocal.last_dim]))
            grad_dul = torch.mean(torch.abs(net.convlocal.layer4[-1].conv2.weight.grad[net.convlocal.last_dim:]))
            optimzer.step()

            timeStr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print(expid,'{:0>3d}-{:0>6d} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(epoch, i, timeStr, loss.item(),
                                                                                        loss_ce.item(),
                                                                                        loss_color.item(),
                                                                                        neg_loss.item(),
                                                                                        loss_dul.item(),
                                                                                        ))
            grad_info_str = '{:.5f},{:.5f},{:.5f},{:.7f},{:.7f}\n'.format(loss_ce.item(),
                                                                          loss_color.item(),
                                                                          loss_dul.item(),
                                                                          grad_color.item(),
                                                                          grad_dul.item())
            grad_info_lst.append(grad_info_str)

            if i % 50 == 0:
                for idx in range(cfg.DATASET.VIDEO_LEN):
                    img1 = frames1[0, idx, :].permute(1, 2, 0).cpu().numpy()
                    img2 = frames2[0, idx, :].permute(1, 2, 0).cpu().numpy()
                    mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                    std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                    img1 = np.array((img1 * std_ + mean_) * 255, dtype=np.uint8)
                    img2 = np.array((img2 * std_ + mean_) * 255, dtype=np.uint8)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    cv2.imwrite('./images/COCO_frame2_{}.jpg'.format(idx), img2)
                    cv2.imwrite('./images/COCO_frame1_{}.jpg'.format(idx), img1)

                for idx in range(cfg.DATASET.VIDEO_LEN - 1):
                    gt1 = labs1[0, idx + 1].detach().cpu().numpy().transpose(1, 2, 0).reshape(height, width, 3) * 255
                    gt1 = np.array(gt1, dtype=np.uint8)

                    color1 = color_lst[idx][0].detach().cpu().numpy().transpose(1, 2, 0).reshape(height, width, 3) * 255
                    color1 = np.array(color1, dtype=np.uint8)

                    cv2.imwrite('./images/COCO_img{}_gt.jpg'.format(idx), gt1)
                    cv2.imwrite('./images/COCO_img{}_pred.jpg'.format(idx), color1)

        with open(r'./snapshots/{}.txt'.format(expid),'a') as fpw:
            fpw.writelines(grad_info_lst)

        if epoch % 20 == 0 or epoch > 290:
            net_sd = net.convlocal.state_dict()
            save_dict = {
                'model': net_sd,
                'train_config': train_config
            }
            torch.save(save_dict,
                           './snapshots/{}_{:0>3d}_{}.pkl'.format(expid,epoch,model_size))



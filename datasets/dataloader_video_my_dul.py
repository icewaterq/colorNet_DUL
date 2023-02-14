"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import random
import torch
import math
import glob

from PIL import Image,ImageFilter,ImageOps
import numpy as np

from .dataloader_base import DLBase
import datasets.daugm_video as tf
import datasets.daugm_video_my as tf_my
import cv2
from torchvision import transforms
import torch.nn.functional as F

def datanorm(x):
    minv = np.min(x)
    maxv = np.max(x)
    if maxv - minv < 1 / 256:
        return x
    return (x - minv) / (maxv - minv)

def dataclip(x):
    mean = np.mean(x)
    std = np.std(x)
    return np.clip(x,mean-std*3,mean+std*3)



class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def addFakeShadow(img, n=2):
    h, w, _ = img.shape
    sizew = int(w * 0.5)
    sizeh = int(h * 0.5)
    for _ in range(n):
        imggamma = img.astype(np.float32)
        mask = np.zeros(img.shape)
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = np.clip(random.randint(x1 - sizew, x1 + sizew), 1, w - 1)
        y2 = np.clip(random.randint(y1 - sizeh, y1 + sizeh), 1, h - 1)
        x3 = random.randint(0, w)
        y3 = random.randint(0, h)
        x4 = np.clip(random.randint(x3 - sizew, x3 + sizew), 1, w - 1)
        y4 = np.clip(random.randint(y3 - sizeh, y3 + sizeh), 1, h - 1)
        pts = np.array([
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4],
        ])
        cv2.fillConvexPoly(mask, pts, (255, 255, 255))
        if (np.max(imggamma) > 0.01):
            imggamma = (imggamma - np.min(imggamma)) / np.max(imggamma)
        if random.random() < 0.5:
            gamma = 0.5 + 0.5 * random.random()
        else:
            gamma = 1 + random.random() * 0.5
        imggamma = ((imggamma ** gamma) * 255).astype(np.uint8)

        if random.random() < 0.8:
            gaussSize = random.choice([3, 5, 7, 9])
            gaussRadius = random.randint(7, 12)
            maskGaus = cv2.GaussianBlur(mask, (gaussSize, gaussSize), gaussRadius)
            maskGaus = maskGaus.astype(np.float32) / 255
            img = imggamma * maskGaus + img * (1 - maskGaus)
        else:
            img[mask > 127] = imggamma[mask > 127]
    return img

def shadow(img):
    h, w, _ = img.shape
    imgarr = np.array(img, dtype=np.float32)
    imgarr = imgarr.swapaxes(1, 2).swapaxes(0, 1)

    if random.random() < 0.5:
        tmp = np.linspace(0, np.pi * (random.random() * 3 + 0.5), num=w) + random.random() * np.pi * 2
        tmp = np.sin(tmp) * 0.35 + 0.85
        tmp = np.clip(tmp, 0.5, 1.2)
        grad = np.array([tmp for i in range(h)])
    else:
        tmp = np.linspace(0, np.pi * (random.random() * 3 + 0.5), num=h) + random.random() * np.pi * 2
        tmp = np.sin(tmp) * 0.35 + 0.85
        tmp = np.clip(tmp, 0.5, 1.2)
        grad = np.array([tmp for i in range(w)])
        grad = grad.swapaxes(0, 1)

    imgarr = (imgarr - np.min(imgarr)) / np.max(imgarr)

    offset = np.mean(imgarr) - 0.5
    grad = grad + offset
    grad = np.clip(grad, 0.5, 1.2)

    imgarr[0] = imgarr[0] ** grad
    imgarr[1] = imgarr[1] ** grad
    imgarr[2] = imgarr[2] ** grad

    imgarr = imgarr * 255
    imgarr = imgarr.swapaxes(0, 1).swapaxes(1, 2).astype(np.uint8)
    return imgarr

def randomMask(img,stride,p):
    h,w,c = img.shape
    img = img.reshape(h//stride, stride, w//stride, stride, c).transpose(0,2,1,3,4).reshape(-1,stride * stride, c)
    patch_num = img.shape[0]
    mask_idx = np.random.randint(0,patch_num,int(patch_num*p))
    img[mask_idx] = np.mean(img[mask_idx],1).reshape(-1,1,c).astype(np.uint8)
    img = img.reshape(h//stride, w//stride, stride, stride, c).transpose(0,2,1,3,4).reshape(h,w,c)
    return img

def randomMaskTensor(img,stride,p):
    c,h,w = img.size()
    img = img.view(c, h//stride, stride, w//stride, stride).permute(1,3,2,4,0).reshape(-1,stride * stride, c)
    patch_num = img.shape[0]
    mask_idx = np.random.randint(0,patch_num,int(patch_num*p))
    img[mask_idx] = -1
    img = img.view(h//stride, w//stride, stride, stride, c).permute(4,0,2,1,3).reshape(c,h,w)
    return img

class AddFakeShadow(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.asarray(img).astype(np.float32)
            img = addFakeShadow(img,random.randint(3,6))
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            return img
        else:
            return img


class AddRandomMask(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.asarray(img).astype(np.uint8)
            stride_lst = [28,56]
            random.shuffle(stride_lst)
            for stride in stride_lst[:2]:
                img = randomMask(img, stride, random.random() * 0.15 + 0.15)  # 0.1 - 0.3
            img = Image.fromarray(img)
            return img
        else:
            return img


mX,mY = np.meshgrid(np.arange(0, 256), np.arange(0, 256))
mX = mX.astype(np.float32)
mY = mY.astype(np.float32)

def generateMap(w,h):
    mapX,mapY = np.meshgrid(np.arange(0, w), np.arange(0, h))
    mapX = mapX.astype(np.float32)
    mapY = mapY.astype(np.float32)
    markPoints = [[random.randint(-int(0.2 * w), int(1.2 * w)), random.randint(-int(0.2 * h), int(1.2 * h))] for _ in range(8)]
    for mp in markPoints:
        cx, cy = mp
        ox, oy = random.randint(12, 24)*random.choice([1,-1]), random.randint(12,24)*random.choice([1,-1])
        dis = ((mX - cx) ** 2 + (mY - cy) ** 2) ** 0.5
        ratio = np.clip(1 - dis / 100, 0, 1)
        mapX += ox * ratio
        mapY += oy * ratio

    grid = np.array([mapX / w * 2 - 1, mapY / h * 2 - 1]).swapaxes(0, 1).swapaxes(1, 2)
    grid = torch.Tensor(grid)
    grid_8 = np.array([cv2.resize(mapX,(0,0),fx = 1/8,fy = 1/8) / w * 2 - 1, cv2.resize(mapY,(0,0),fx = 1/8,fy = 1/8) / h * 2 - 1]).swapaxes(0, 1).swapaxes(1, 2)
    grid_8 = torch.Tensor(grid_8)
    return mapX,mapY,grid,grid_8


def findEdge2(img,lenThresh,stride,scale = 1):
    img = cv2.resize(img,(0,0),fx=scale,fy=scale)
    img = cv2.GaussianBlur(img,(5,5),3)
    mask = cv2.Canny(img,100,200)
    h,w = mask.shape
    mask[:int(h * 0.1)] = 0
    mask[int(h * 0.9):] = 0
    mask[:,:int(w * 0.1)] = 0
    mask[:,int(w * 0.9):] = 0
    mask2 = np.zeros((h,w,3),dtype=np.uint8)

    for i in range(0,w,stride):
        length = 0
        lastV = 0
        for j in range(h):
            curV = mask[j,i]
            if lastV>127 and curV<127:
                length = 0
            if lastV<127 and curV<127:
                length+=1
            if lastV<127 and curV>127:
                if length > lenThresh:
                    mask2[j-4:j+4, i-4:i+4] = 255
                length = 0
            lastV = curV

    for i in range(0,w,stride):
        length = 0
        lastV = 0
        for j in range(h):
            j = h-j-1
            curV = mask[j,i]
            if lastV>127 and curV<127:
                length = 0
            if lastV<127 and curV<127:
                length+=1
            if lastV<127 and curV>127:
                if length > lenThresh:
                    mask2[j-4:j+4, i-4:i+4] = 255
                length = 0
            lastV = curV


    for j in range(0,h,stride):
        length = 0
        lastV = 0
        for i in range(w):
            curV = mask[j,i]
            if lastV>127 and curV<127:
                length = 0
            if lastV<127 and curV<127:
                length+=1
            if lastV<127 and curV>127:
                if length > lenThresh:
                    mask2[j-4:j+4, i-4:i+4] = 255
                length = 0
            lastV = curV

    for j in range(0,h,stride):
        length = 0
        lastV = 0
        for i in range(w):
            i = w - i - 1
            curV = mask[j,i]
            if lastV>127 and curV<127:
                length = 0
            if lastV<127 and curV<127:
                length+=1
            if lastV<127 and curV>127:
                if length > lenThresh:
                    mask2[j-4:j+4, i-4:i+4] = 255
                length = 0
            lastV = curV

    mask2 = cv2.resize(mask2,(0,0),fx = 1/scale,fy = 1/scale)
    mask2 = (mask2.astype(np.float32)/255) * 4 + 1
    return mask2


class DataVideo(DLBase):

    def __init__(self, cfg, split, val=False):
        super(DataVideo, self).__init__()

        self.cfg = cfg
        self.split = split
        self.val = val

        self.cfg_frame_gap = cfg.DATASET.VAL_FRAME_GAP if val else cfg.DATASET.FRAME_GAP

        self._init_palette(cfg.TRAIN.BATCH_SIZE * cfg.MODEL.GRID_SIZE**2)

        # train/val/test splits are pre-cut
        split_fn = os.path.join(cfg.DATASET.ROOT, "filelists", self.split + ".txt")
        print('split_fn',split_fn)
        assert os.path.isfile(split_fn)

        self.images = []

        token = None # sequence token (new video when it changes)

        subsequence = []
        ignored = [0]
        num_frames = [0]

        def add_sequence():

            if cfg.DATASET.VIDEO_LEN > len(subsequence):
                # found a very short sequence
                ignored[0] += 1
            else:
                # adding the subsequence
                self.images.append(tuple(subsequence))
                num_frames[0] += len(subsequence)

            del subsequence[:]

        with open(split_fn, "r") as lines:
            for line in lines:
                _line = line.strip("\n").split(' ')

                assert len(_line) > 0, "Expected at least one path"
                _image = _line[0]

                # each sequence may have a different length
                # do some book-keeping e.g. to ensure we have
                # sequences long enough for subsequent sampling
                _token = _image.split("/")[-2] # parent directory

                # sequence ID is in the filename
                #_token = os.path.basename(_image).split("_")[0]
                if token != _token:
                    if not token is None:
                        add_sequence()
                    token = _token

                # image 1
                _image = os.path.join(cfg.DATASET.ROOT, _image.lstrip('/'))
                #assert os.path.isfile(_image), '%s not found' % _image
                subsequence.append(_image)

        # update the last sequence
        # returns the total amount of frames
        add_sequence()
        print("Dataloader: {}".format(split), " / Frame Gap: ", self.cfg_frame_gap)
        print("Loaded {} sequences / {} ignored / {} frames".format(len(self.images), \
                                                                    ignored[0], \
                                                                    num_frames[0]))

        self._num_samples = num_frames[0]
        self._init_augm(cfg)
        self._init_augm_my(cfg)

        self.color_jitter_l = transforms.Compose([
            AddFakeShadow(0.6),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.2),
        ])

        self.color_jitter_s = transforms.Compose([
            AddFakeShadow(0.6),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.3),
            Solarization(0.1),
        ])
        self.random_mask = transforms.Compose([
            AddRandomMask(0.5)
        ])

    def _init_augm(self, cfg):

        # general (unguided) affine transformations
        tfs_pre = [tf.CreateMask()]
        self.tf_pre = tf.Compose(tfs_pre)

        # photometric noise
        tfs_affine = []

        # guided affine transformations
        tfs_augm = []

        # 1.
        # general affine transformations
        #
        tfs_pre.append(tf.MaskScaleSmallest(cfg.DATASET.SMALLEST_RANGE))

        if cfg.DATASET.RND_CROP:
            tfs_pre.append(tf.MaskRandCrop(cfg.DATASET.CROP_SIZE, pad_if_needed=True))
        else:
            tfs_pre.append(tf.MaskCenterCrop(cfg.DATASET.CROP_SIZE))

        if cfg.DATASET.RND_HFLIP:
            tfs_pre.append(tf.MaskRandHFlip())

        # 2.
        # Guided affine transformation
        #
        if cfg.DATASET.GUIDED_HFLIP:
            tfs_affine.append(tf.GuidedRandHFlip())

        # this will add affine transformation
        if cfg.DATASET.RND_ZOOM:
            tfs_affine.append(tf.MaskRandScaleCrop(*cfg.DATASET.RND_ZOOM_RANGE))

        self.tf_affine = tf.Compose(tfs_affine)
        self.tf_affine2 = tf.Compose([tf.AffineIdentity()])

        tfs_post = [tf.ToTensorMask(),
                    tf.Normalize(mean=self.MEAN, std=self.STD),
                    tf.ApplyMask(-1)]

        # image to the teacher will have no noise
        self.tf_post = tf.Compose(tfs_post)

    def _init_augm_my(self, cfg):

        # general (unguided) affine transformations
        tfs_pre = [tf_my.CreateMask()]
        self.tf_pre_my = tf_my.Compose(tfs_pre)

        # photometric noise
        tfs_affine = []

        # guided affine transformations
        tfs_augm = []

        # 1.
        # general affine transformations
        #
        tfs_pre.append(tf_my.MaskScaleSmallest(cfg.DATASET.SMALLEST_RANGE))
        if cfg.DATASET.RND_CROP:
            tfs_pre.append(tf_my.MaskRandCrop_My(cfg.DATASET.CROP_SIZE, pad_if_needed=True))
        else:
            tfs_pre.append(tf_my.MaskCenterCrop(cfg.DATASET.CROP_SIZE))

        if cfg.DATASET.RND_HFLIP:
            tfs_pre.append(tf_my.MaskRandHFlip())

        # 2.
        # Guided affine transformation
        #
        if cfg.DATASET.GUIDED_HFLIP:
            tfs_affine.append(tf_my.GuidedRandHFlip())

        # this will add affine transformation
        if cfg.DATASET.RND_ZOOM:
            tfs_affine.append(tf_my.MaskRandScaleCrop(*cfg.DATASET.RND_ZOOM_RANGE))

        self.tf_affine_my = tf_my.Compose(tfs_affine)
        self.tf_affine2_my = tf_my.Compose([tf_my.AffineIdentity()])

        tfs_post = [tf_my.ToTensorMask(),
                    tf_my.Normalize(mean=self.MEAN, std=self.STD),
                    tf_my.ApplyMask(-1)]

        # image to the teacher will have no noise
        self.tf_post_my = tf_my.Compose(tfs_post)


    def set_num_samples(self, n):
        print("Re-setting # of samples: {:d} -> {:d}".format(self._num_samples, n))
        self._num_samples = n

    def __len__(self):
        return len(self.images) #self._num_samples

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image

    def _get_affine(self, params):

        N = len(params)

        # construct affine operator
        affine = torch.zeros(N, 2, 3)

        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / \
                            float(self.cfg.DATASET.CROP_SIZE[1])

        for i, (dy,dx,alpha,scale,flip) in enumerate(params):

            # R inverse
            sin = math.sin(alpha * math.pi / 180.)
            cos = math.cos(alpha * math.pi / 180.)

            # inverse, note how flipping is incorporated
            affine[i,0,0], affine[i,0,1] = flip * cos, sin * aspect_ratio
            affine[i,1,0], affine[i,1,1] = -sin / aspect_ratio, cos

            # T inverse Rinv * t == R^T * t
            affine[i,0,2] = -1. * (cos * dx + sin * dy)
            affine[i,1,2] = -1. * (-sin * dx + cos * dy)

            # T
            affine[i,0,2] /= float(self.cfg.DATASET.CROP_SIZE[1] // 2)
            affine[i,1,2] /= float(self.cfg.DATASET.CROP_SIZE[0] // 2)

            # scaling
            affine[i] *= scale

        return affine

    def _get_affine_inv(self, affine, params):

        aspect_ratio = float(self.cfg.DATASET.CROP_SIZE[0]) / \
                            float(self.cfg.DATASET.CROP_SIZE[1])

        affine_inv = affine.clone()
        affine_inv[:,0,1] = affine[:,1,0] * aspect_ratio**2
        affine_inv[:,1,0] = affine[:,0,1] / aspect_ratio**2
        affine_inv[:,0,2] = -1 * (affine_inv[:,0,0] * affine[:,0,2] + affine_inv[:,0,1] * affine[:,1,2])
        affine_inv[:,1,2] = -1 * (affine_inv[:,1,0] * affine[:,0,2] + affine_inv[:,1,1] * affine[:,1,2])

        # scaling
        affine_inv /= torch.Tensor(params)[:,3].view(-1,1,1)**2

        return affine_inv

    def __getitem__(self, index):
        isTrans = True
        if random.random()<0.5:
            isTrans = False

        # searching for the video clip ID
        sequence = self.images[index] # % len(self.images)]
        seqlen = len(sequence)

        assert self.cfg_frame_gap > 0, "Frame gap should be positive"
        t_window = self.cfg_frame_gap * self.cfg.DATASET.VIDEO_LEN

        # reduce sampling gap for short clips
        t_window = min(seqlen, t_window)
        frame_gap = t_window // self.cfg.DATASET.VIDEO_LEN

        # strided slice
        frame_ids = torch.arange(t_window)[::frame_gap]
        frame_ids = frame_ids[:self.cfg.DATASET.VIDEO_LEN]
        assert len(frame_ids) == self.cfg.DATASET.VIDEO_LEN

        # selecting a random start
        index_start = random.randint(0, seqlen - frame_ids[-1] - 1)
        # permuting the frames in the batch
        random_ids = torch.randperm(self.cfg.DATASET.VIDEO_LEN)
        # adding the offset
        frame_ids = frame_ids[random_ids] + index_start

        # print('frame_ids',frame_ids)

        # forward sequence
        images = []
        images_dul = []
        for frame_id in frame_ids:
            fn = sequence[frame_id]
            img = Image.open(fn).convert('RGB')
            images_dul.append(img)
            if random.random()<0.3:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            images.append(img)


        #===================================my=================================
        frames, valid = self.tf_pre_my(images)
        frames1, valid1 = frames[:], valid[:]
        frames2 = [f.copy() for f in frames]
        valid2 = [v.copy() for v in valid]
        frames1, valid1, affine_params1 = self.tf_affine_my(frames1, valid1)
        frames2, valid2, affine_params2 = self.tf_affine2_my(frames2, valid2)
        labs1 = []
        labs2 = []
        rgbs1 = []
        rgbs2 = []
        for i in range(self.cfg.DATASET.VIDEO_LEN):
            lab1 = cv2.cvtColor(np.asarray(frames1[i]), cv2.COLOR_RGB2LAB)
            lab2 = cv2.cvtColor(np.asarray(frames2[i]), cv2.COLOR_RGB2LAB)
            labs1.append(lab1)
            labs2.append(lab2)

            rgbs1.append(np.asarray(frames1[i]))
            rgbs2.append(np.asarray(frames2[i]))

        labs1 = np.array(labs1, dtype=np.float32).transpose(0,3,1,2) / 255
        labs2 = np.array(labs2, dtype=np.float32).transpose(0,3,1,2) / 255

        rgbs1 = np.array(rgbs1, dtype=np.float32).transpose(0, 3, 1, 2) / 255
        rgbs2 = np.array(rgbs2, dtype=np.float32).transpose(0, 3, 1, 2) / 255

        labs1[:,1:] = dataclip(labs1[:,1:])
        labs2[:,1:] = dataclip(labs2[:,1:])
        labs1[:,1:] = datanorm(labs1[:,1:])
        labs2[:,1:] = datanorm(labs2[:,1:])

        labs1 = torch.from_numpy(labs1)
        labs2 = torch.from_numpy(labs2)
        rgbs1 = torch.from_numpy(rgbs1)
        rgbs2 = torch.from_numpy(rgbs2)


        w_mask_lst = []
        for i in range(self.cfg.DATASET.VIDEO_LEN):
            if isTrans:
                frames1[i] = self.color_jitter_s(frames2[i])
            else:
                frames1[i] = self.color_jitter_s(frames1[i])
            frames2[i] = self.color_jitter_l(frames2[i])
            w_mask = findEdge2(np.asarray(frames2[i]), 16, 16, scale=0.5)
            w_mask_lst.append(w_mask)

        w_mask_lst = np.array(w_mask_lst) #T,H,W,3
        w_mask_lst = w_mask_lst.transpose(0,3,1,2)
        w_mask_lst = torch.from_numpy(w_mask_lst)


        # convert to tensor, zero out the values

        frames1 = self.tf_post_my(frames1, valid1)
        frames2 = self.tf_post_my(frames2, valid2)

        # converting the affine transforms
        aff_reg = self._get_affine(affine_params1)
        aff_main = self._get_affine(affine_params2)

        aff_reg_inv = self._get_affine_inv(aff_reg, affine_params1)

        aff_reg = aff_main # identity affine2_inv
        aff_main = aff_reg_inv

        frames1 = torch.stack(frames1, 0)
        frames2 = torch.stack(frames2, 0)

        aff_main = F.affine_grid(aff_main, size=(frames1.size(0),self.cfg.MODEL.FEATURE_DIM+1000,frames1.size(2)//8,frames1.size(3)//8), align_corners=False)
        aff_reg = F.affine_grid(aff_reg, size=(frames1.size(0),self.cfg.MODEL.FEATURE_DIM,frames1.size(2)//8,frames1.size(3)//8), align_corners=False)
        if isTrans:
            # print('aff_main',aff_main.size())
            # _,_,grid = generateMap(frames1.size(3),frames1.size(2))
            aff_main = []
            aff_main_8 = []
            for i in range(self.cfg.DATASET.VIDEO_LEN):
                grid, grid_8 = generateMap(frames1.size(3), frames1.size(2))[2:]
                aff_main.append(grid)
                aff_main_8.append(grid_8)

            aff_main = torch.stack(aff_main)
            aff_main_8 = torch.stack(aff_main_8)

            frames1 = F.grid_sample(frames1, aff_main, align_corners=False, mode="bilinear",padding_mode='reflection')
            labs1 = F.grid_sample(labs2, aff_main, align_corners=False, mode="bilinear", padding_mode='reflection')

        assert frames1.shape == frames2.shape
        if isTrans:
            aff_main = aff_main_8
        # ===================================my=================================

        # ===================================dul=================================
        frames_dul, valid_dul = self.tf_pre(images_dul)

        frames1_dul, valid1_dul = frames_dul[:], valid_dul[:]
        frames2_dul = [f.copy() for f in frames_dul]
        valid2_dul = [v.copy() for v in valid_dul]

        frames1_dul, valid1_dul, affine_params1_dul = self.tf_affine(frames1_dul, valid1_dul)
        frames2_dul, valid2_dul, affine_params2_dul = self.tf_affine2(frames2_dul, valid2_dul)


        frames1_dul = self.tf_post(frames1_dul, valid1_dul)
        frames2_dul = self.tf_post(frames2_dul, valid2_dul)

        aff_reg_dul = self._get_affine(affine_params1_dul)
        aff_main_dul = self._get_affine(affine_params2_dul)

        aff_reg_inv_dul = self._get_affine_inv(aff_reg_dul, affine_params1_dul)
        aff_reg_dul = aff_main_dul
        aff_main_dul = aff_reg_inv_dul

        frames1_dul = torch.stack(frames1_dul, 0)
        frames2_dul = torch.stack(frames2_dul, 0)

        # ===================================dul=================================

        return frames2, frames1, aff_main, aff_reg, labs2, labs1, rgbs2, rgbs1, w_mask_lst, frames2_dul, frames1_dul, aff_main_dul, aff_reg_dul

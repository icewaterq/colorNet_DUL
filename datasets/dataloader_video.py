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
import cv2
from torchvision import transforms

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


class DataVideo(DLBase):

    def __init__(self, cfg, split, val=False, is_aug = False):
        super(DataVideo, self).__init__()

        self.cfg = cfg
        self.split = split
        self.val = val

        self.cfg_frame_gap = cfg.DATASET.VAL_FRAME_GAP if val else cfg.DATASET.FRAME_GAP

        self._init_palette(cfg.TRAIN.BATCH_SIZE * cfg.MODEL.GRID_SIZE**2)

        # train/val/test splits are pre-cut
        split_fn = os.path.join(cfg.DATASET.ROOT, "filelists", self.split + ".txt")
        assert os.path.isfile(split_fn)
        self.is_aug = is_aug
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
        
        self.color_jitter_l = transforms.Compose([
            AddFakeShadow(0.5),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.2),
            Solarization(0.1),
        ])

        self.color_jitter_s = transforms.Compose([
            AddFakeShadow(0.5),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.3),
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

        # forward sequence
        images = []
        for frame_id in frame_ids:
            fn = sequence[frame_id]
            images.append(Image.open(fn).convert('RGB'))

        # 1. general transforms
        frames, valid = self.tf_pre(images)

        # 1.1 creating two sequences in forward/backward order
        frames1, valid1 = frames[:], valid[:]

        # second copy
        frames2 = [f.copy() for f in frames]
        valid2 = [v.copy() for v in valid]

        # 2. guided affine transforms
        frames1, valid1, affine_params1 = self.tf_affine(frames1, valid1)
        frames2, valid2, affine_params2 = self.tf_affine2(frames2, valid2)
        
        if self.is_aug:
            #print('aug data.')
            for i in range(self.cfg.DATASET.VIDEO_LEN):
                frames1[i] = self.color_jitter_s(frames1[i])
                frames2[i] = self.color_jitter_l(frames2[i])

        # convert to tensor, zero out the values
        frames1 = self.tf_post(frames1, valid1)
        frames2 = self.tf_post(frames2, valid2)

        # converting the affine transforms
        aff_reg = self._get_affine(affine_params1)
        aff_main = self._get_affine(affine_params2)

        aff_reg_inv = self._get_affine_inv(aff_reg, affine_params1)

        aff_reg = aff_main # identity affine2_inv
        aff_main = aff_reg_inv

        frames1 = torch.stack(frames1, 0)
        frames2 = torch.stack(frames2, 0)

        assert frames1.shape == frames2.shape

        return frames2, frames1, aff_main, aff_reg

class DataVideoKinetics(DataVideo):

    def __init__(self, cfg, split):
        super(DataVideo, self).__init__()

        self.cfg = cfg
        self.split = split

        self._init_palette(cfg.TRAIN.BATCH_SIZE * cfg.MODEL.GRID_SIZE**2)

        # train/val/test splits are pre-cut
        split_fn = os.path.join(cfg.DATASET.ROOT, "filelists", self.split + ".txt")
        assert os.path.isfile(split_fn)

        self.videos = []

        with open(split_fn, "r") as lines:
            for line in lines:
                _line = line.strip("\n").split(' ')
                assert len(_line) > 0, "Expected at least one path"

                _vid = _line[0]
    
                # image 1
                _vid = os.path.join(cfg.DATASET.ROOT, _vid.lstrip('/'))
                #assert os.path.isdir(_vid), "{} does not exist".format(_vid)
                self.videos.append(_vid)

        # update the last sequence
        # returns the total amount of frames
        print("DataloaderKinetics: {}".format(split))
        print("Loaded {} sequences".format(len(self.videos)))
        self._init_augm(cfg)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):

        C = self.cfg.DATASET
        path = self.videos[index]

        # filenames
        fns = sorted(glob.glob(path + "/*.jpg"))
        total_len = len(fns)

        # temporal window to consider
        temp_window = min(total_len, C.VIDEO_LEN * C.FRAME_GAP)
        gap = temp_window / C.VIDEO_LEN
        start_frame = random.randint(0, total_len - temp_window)

        images = []
        for idx in range(C.VIDEO_LEN):
            frame_id = start_frame + int(idx * gap)
            fn = fns[frame_id]
            images.append(Image.open(fn).convert('RGB'))

        # 1. general transforms
        frames, valid = self.tf_pre(images)

        # 1.1 creating two sequences in forward/backward order
        frames1, valid1 = frames[:], valid[:]

        # second copy
        frames2 = [f.copy() for f in frames]
        valid2 = [v.copy() for v in valid]

        # 2. guided affine transforms
        frames1, valid1, affine_params1 = self.tf_affine(frames1, valid1)
        frames2, valid2, affine_params2 = self.tf_affine2(frames2, valid2)

        # convert to tensor, zero out the values
        frames1 = self.tf_post(frames1, valid1)
        frames2 = self.tf_post(frames2, valid2)

        # converting the affine transforms
        affine1 = self._get_affine(affine_params1)
        affine2 = self._get_affine(affine_params2)

        affine1_inv = self._get_affine_inv(affine1, affine_params1)

        affine1 = affine2 # identity affine2_inv
        affine2 = affine1_inv

        frames1 = torch.stack(frames1, 0)
        frames2 = torch.stack(frames2, 0)

        assert frames1.shape == frames2.shape

        return frames2, frames1, affine2, affine1

import os
from os.path import join,basename
import numpy as np
import imageio
import time
from models.resnet_my import resnet18

import torch.multiprocessing as mp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import argparse
from glob import glob


if __name__=='__main__':

    modelLst=[
        'color_len5_300_L.pkl',
        'color_len4_300_L.pkl',
        'color_len3_300_L.pkl',
        'color_len2_300_L.pkl',
        'color_noaug_300_L.pkl',
        'color_noedge_300_L.pkl',
        'color_nolabclip_300_L.pkl',
        'color_noliquid_300_L.pkl',
        'color_noneg_300_L.pkl',
        'color_nocs_300_L.pkl',
        'color_noshadow_300_L.pkl',
        'color_rgb_300_L.pkl',
        'color_dul_300_M.pkl',
        'color_dul_300_S.pkl',
        'color_dul_coco_300_M.pkl',
        'color_dul_10_300_M.pkl',
        'color_dul_100_300_M.pkl',
    ]

    for modelname in modelLst:

        merge_sd={}
        count = 0
        for i in range(290,300):
            count+=1
            model_path = join(r'E:\paper\实验记录',modelname.replace('300',str(i+1)))
            print('load model : {}'.format(model_path))
            model_pkl = torch.load(model_path)
            net_sd = model_pkl['model']
            train_config = model_pkl['train_config']

            for key in net_sd:
                if 'num_batches_tracked' in key:
                    continue
                if key not in merge_sd:
                    merge_sd[key] = net_sd[key]
                else:
                    merge_sd[key] += net_sd[key]
        for key in merge_sd:
            if 'num_batches_tracked' in key:
                continue
            merge_sd[key] /= count
        save_dict = {
            'model': merge_sd,
            'train_config': train_config
        }
        torch.save(save_dict,join(r'./snapshots/{}'.format(modelname.replace('300','merge'))))
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

stride = 8
width = 864
height = 480


def img2RGB_LAB(path):

    MEAN = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    STD = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = cv2.imread(path)
    img = cv2.resize(img,(width,height))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    rgb = rgb.transpose(2,0,1).astype(np.float32)
    lab = lab.transpose(2,0,1).astype(np.float32)
    lab = lab.reshape(3, int(height//stride), stride, int(width//stride), stride).transpose(0,2,4,1,3).reshape(-1, int(height//stride), int(width//stride))

    rgb = (rgb/255-MEAN)/STD
    lab = lab/255
    rgb = torch.from_numpy(rgb).float().cuda().unsqueeze(0)
    lab = torch.from_numpy(lab).float().cuda().unsqueeze(0)
    return rgb,lab

def colorize(feats, labs):
    color_pred_lst = []
    # feats:(B,C,H,W)
    # labs:(B,C,H,W)
    B,C,H,W = feats.size()
    feat1 = feats[0]
    lab1 = labs[0]
    feat1 = feat1.view(C,H*W).permute(1,0)  #HW,C
    lab1 = lab1.view(-1,H*W).permute(1,0)    #HW,3
    lossLst = []
    for i in range(1,B):
        feat2 = feats[i]
        feat2 = feat2.view(C, H * W).permute(1, 0)  #HW,C
        lab2 = labs[i]
        lab2 = lab2.view(-1, H * W).permute(1, 0)

        att1 = torch.mm(feat2, feat1.permute(1, 0)) * 25
        att1 = F.softmax(att1, dim=1)  # [HW,HW]

        lab2_pred = torch.mm(att1, lab1)
        loss_color= F.l1_loss(lab2, lab2_pred)
        lossLst.append(loss_color.item())

        # img = lab2.view(H, W, 3, stride, stride).permute(0, 3, 1, 4, 2).reshape(height, width, 3)
        # img_pred = lab2_pred.view(H, W, 3, stride, stride).permute(0, 3, 1, 4, 2).reshape(height, width, 3)
        # img = (img.cpu().numpy()*255).astype(np.uint8)
        # img_pred = (img_pred.cpu().numpy()*255).astype(np.uint8)
        # cv2.imshow('img',img)
        # cv2.imshow('img_pred',img_pred)
        # cv2.waitKey()


        # color_pred_lst.append(img_pred)
    return np.mean(lossLst)

def eval_dir(path,model):
    flist = os.listdir(path)
    num = len(flist)
    featLst = []
    labLst = []
    for f in flist:
        rgb,lab = img2RGB_LAB(join(path,f))
        q_color, _, color, _ = model(rgb)
        featLst.append(q_color)
        labLst.append(lab)
    featLst = torch.cat(featLst)
    labLst = torch.cat(labLst)
    lossLst = []
    for i in range(num-31):
        # loss = colorize(featLst[i:i+21], labLst[i:i+21])
        loss = colorize(featLst[[i,i+15,i+30]], labLst[[i,i+15,i+30]])
        lossLst.append(loss)
    return np.mean(lossLst)


val_name = [
'bear',
'bmx-bumps',
'boat',
'boxing-fisheye',
'breakdance-flare',
'bus',
'car-turn',
'cat-girl',
'classic-car',
'color-run',
'dancing',
'disc-jockey',
'dog-gooses',
'dogs-scale',
'drift-turn',
'drone',
'elephant',
'flamingo',
'hike',
'hockey',
'kid-football',
'kite-walk',
'koala',
'lady-running',
'lindy-hop',
'lucia',
'mallard-fly',
'mallard-water',
'miami-surf',
'paragliding',
'rhino',
'schoolgirls',
'scooter-board',
'scooter-gray',
'sheep',
'skate-park',
'snowboard',
'stroller',
'stunt',
'tennis',
'tractor-sand',
'train',
'upside-down',
'varanus-cage',
'walking',
]

if __name__=='__main__':
    # dirlist= os.listdir(r'D:\data\DAVIS2017\JPEGImages\480p')
    # flist1 = os.listdir(r'D:\data\DAVIS2017\result\res4_vis')
    # for dir in dirlist:
    #     if dir in flist1:
    #         continue
    #     flist = os.listdir(join(r'D:\data\DAVIS2017\JPEGImages\480p',dir))
    #     if len(flist)>60:
    #         print("'{}',".format(dir))
    #     # print(dir,len(flist))
    lst = [
    ['color_dul_291_M.pkl', 0.0237991],
    ['color_dul_292_M.pkl', 0.0240105],
    ['color_dul_293_M.pkl', 0.0239755],
    ['color_dul_294_M.pkl', 0.0240390],
    ['color_dul_295_M.pkl', 0.0238615],
    ['color_dul_296_M.pkl', 0.0238913],
    ['color_dul_297_M.pkl', 0.0239387],
    ['color_dul_298_M.pkl', 0.0239216],
    ['color_dul_299_M.pkl', 0.0242335],
    ['color_dul_300_M.pkl', 0.0240904],
    ]
    lst.sort(key=lambda x:x[1])
    for name,loss in lst:
        print(name,loss)


    parser = argparse.ArgumentParser('')
    parser.add_argument('--prefix', type=str, help="模型前缀")
    parser.add_argument('--datapath', type=str, help="数据路径")
    args = parser.parse_args()

    dirLst = []
    for name in val_name:
        dirLst.append(join(args.datapath,name))

    modelLst = glob(args.prefix)
    bestLoss = 10000000
    bestModel = 'None'
    for model_path in modelLst:
        print('load model : {}'.format(model_path))
        model_pkl = torch.load(model_path)
        net_sd = model_pkl['model']
        train_config = model_pkl['train_config']

        print('create model.')
        print('train_config')
        print(train_config)

        feat_mode = 'color_dul'
        if not train_config['is_dul']:
            feat_mode = 'color'
        model = resnet18(model_size=train_config['model_size'], first_kernal_size=train_config['first_kernal_size'],
                         feat_mode=feat_mode)

        model.load_state_dict(net_sd, strict=True)

        for p in model.parameters():
            p.requires_grad = False

        # setting the evaluation mode
        model.eval()
        model = model.cuda()
        lossLst = []
        for datapath in tqdm(dirLst):
            loss = eval_dir(datapath,model)
            lossLst.append(loss)
        loss = np.mean(lossLst)
        print('{} : {:.7f}'.format(basename(model_path),loss))
        if loss<bestLoss:
            bestLoss = loss
            bestModel = basename(model_path)
    print('bestModel : {}  {:.7f}'.format(bestModel,bestLoss))
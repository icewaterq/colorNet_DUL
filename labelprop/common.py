"""
Based on the inference routines from Jabri et al., (2020)
Credit: https://github.com/ajabri/videowalk.git
License: MIT
"""

import sys
import torch
from labelprop.crw import MaskedAttention
from labelprop.crw import mem_efficient_batched_affinity as batched_affinity

class LabelPropVOS(object):

    def context_long(self):
        """Returns indices of the timesteps
        for long-term memory
        """
        raise NotImplementedError()

    def context_short(self, t):
        """
        Args:
            t: current timestep
        Returns:
            list: indices of timesteps
                  for the context
        """
        raise NotImplementedError()
    
    def predict(self, feats, masks, curr_feat):
        """
        Args:
            feats [C,K,h,w]: context features
            masks [C,M,h,w]: context masks
            curr_feat [1,K,h,w]: current frame features
        Returns:
            mask [1,M,h,w]: current frame mask
        """
        raise NotImplementedError()


class LabelPropVOS_CRW(LabelPropVOS):

    def __init__(self, cfg):
        self.cxt_size = cfg.TEST.CXT_SIZE
        self.radius = cfg.TEST.RADIUS
        self.temperature = cfg.TEST.TEMP
        self.topk = cfg.TEST.KNN
        self.mask = None
        self.mask_hw = None
        print('+++++++ infer radius : {}'.format(self.radius))

    def context_long(self, t0, t):
        return [t0]

    def context_short(self, t0, t):
        to_t = t
        from_t = to_t - self.cxt_size
        timesteps = [max(tt, t0) for tt in range(from_t, to_t)]
        return timesteps

    def context_short_custom_interval(self, t0, t):
        # stride_lst = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 8]
        stride_lst = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 14, 16, 18, 20, 23, 26, 30, 34, 39]
        to_t = t
        from_t = to_t - self.cxt_size
        timesteps = []
        for stride in stride_lst:
            timesteps.insert(0,max(t0 ,to_t - stride))
        return timesteps


    def context_index(self, t0, t, custom_interval = False):
        if custom_interval:
            index_short = self.context_short_custom_interval(t0, t)
        else:
            index_short = self.context_short(t0, t)
        index_long = self.context_long(t0, t)
        cxt_index = index_long + index_short
        return cxt_index

    def predict(self, feats, masks, curr_feat, ref_index=None, t=None):
        """
        Args:
            feats: list of C [1,K,h,w] context features
            masks: list of C [1,M,h,w] context masks
            curr_feat: [1,K,h,w] current frame features
            ref_index: C indices of context frames
            t: current frame time step
        Returns:
            mask [1,M,h,w]: current frame mask
        """
        dev = curr_feat.device
        h, w = curr_feat.shape[-2:]

        # [BC+N,M,h,w] -> [BC+N,h,w,M]
        ctx_lbls = torch.cat(masks, 0).permute([0,2,3,1])

        # keys [1,K,C,h,w]: context features
        keys = torch.stack(feats, 2)[:, :, None]
        # query: [1,K,1,h,w]: reference feature
        query = curr_feat[:, :, None]

        if self.mask is None or self.mask_hw != (h, w):
            # Make spatial radius mask TODO use torch.sparse
            restrict = MaskedAttention(self.radius, flat=False)
            D = restrict.mask(h, w)[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0
            self.mask = D.to(dev)
            self.mask_hw = (h, w)

        # Flatten source frame features to make context feature set
        keys, query = keys.flatten(-2), query.flatten(-2)

        long_mem = [0]
        Ws, Is = batched_affinity(query, keys, self.mask,  \
                self.temperature, self.topk, long_mem)

        # Soft labels of source nodes
        ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

        # Weighted sum of top-k neighbours (Is is index, Ws is weight) 
        pred = (ctx_lbls[:, Is[0].to(dev)] * Ws[0][None].to(dev)).sum(1)
        pred = pred.view(-1, h, w)
        pred = pred.permute(1,2,0)

        # Adding Predictions            
        pred = pred.permute([2,0,1])[None, ...]

        return pred

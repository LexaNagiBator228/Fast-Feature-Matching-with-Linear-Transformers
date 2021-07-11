import argparse
import glob
import numpy as np
import os
import time
import torch.utils.data
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import json
import torch.nn.functional as F

import cv2
import torch

from transformer import LocalFeatureTransformer






class MultiHead_two(nn.Module):
    def __init__(self, f_dim, head_size,d_q,d_v):
        super(MultiHead_two, self).__init__()

        dropout = 0.2


        self.n_head = head_size
        self.d_q = d_q
        self.d_v = d_v
        ###### att_W ######
        #d_q = f_dim//n_head
        self.w_q = nn.Linear(f_dim, f_dim ,bias = False)
        self.w_k = nn.Linear(f_dim, f_dim ,bias = False)
        self.w_v = nn.Linear(f_dim, f_dim ,bias = False)
        self.drop = nn.Dropout(dropout)

        self.fc = nn.Linear(f_dim,f_dim)
        self.fc_drop = nn.Dropout(dropout)


    def forward(self,X,C):
        
        q = self.w_q(X)
        v = self.w_v(C)
        k = self.w_k(C)

        cross_atention = torch.einsum('bmd,bnd->bnm', k,q)/(q.shape[1]**0.5)
        cross_atention = cross_atention.softmax(dim = -1)
        #cross_atention = self.drop(cross_atention)
        res = cross_atention @ v
        res=  self.drop(res)
        res = self.fc(res)


        return res, cross_atention


def MLP(channels: list, do_bn=False, do_bias= True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Linear(channels[i - 1], channels[i], bias=do_bias))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.LeakyReLU(0.2,inplace = True))
            #layers.append(nn.GELU())
    return nn.Sequential(*layers)


class AttentionNet(nn.Module):
    def __init__(self, f_dim, head_size, compute_map=True):
        super(AttentionNet, self).__init__()
        self.w_q = MLP([f_dim, f_dim*2,f_dim])
        self.w_k = MLP([f_dim, f_dim*2,f_dim])
        self.w_v = MLP([f_dim, f_dim*2,f_dim])
    def forward(self, inp, context):
        Q = self.w_q(inp) # BxNxD
        K = self.w_k(context) # BxMxD 
        V = self.w_v(context) # BxMxD

        cross_atention = torch.einsum('bmd,bnd->bnm')

        return cross_atention

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class MultiHead_self(nn.Module):
    def __init__(self, f_dim, head_size,d_q,d_v):
        super(MultiHead_self, self).__init__()

        dropout = 0.2


        self.n_head = head_size
        self.d_q = d_q
        self.d_v = d_v
        ###### att_W ######
        #d_q = f_dim//n_head
        self.w_q = nn.Linear(f_dim, d_q ,bias = False)
        self.w_k = nn.Linear(f_dim, d_q ,bias = False)
        self.w_v = nn.Linear(f_dim, d_v ,bias = False)

        self.drop = nn.Dropout(dropout)

        self.fc = MLP([d_v,d_v, f_dim])
        self.fc_drop = nn.Dropout(dropout)
        self.MHSA = nn.MultiheadAttention(d_v, head_size, dropout= dropout)


    def forward(self,inp):  


        q = self.w_q(inp)
        v = self.w_v(inp)
        k = self.w_k(inp)

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        attn_output, attn_output_weights  = self.MHSA(q,k,v)
        att = attn_output.transpose(0, 1)

        att = self.fc(att)
        att = self.fc_drop(att)

        return att

class MultiHead_with_context(nn.Module):
    def __init__(self, f_dim, head_size,d_q,d_v):
        super(MultiHead_with_context, self).__init__()

        dropout = 0.2


        self.n_head = head_size
        self.d_q = d_q
        self.d_v = d_v
        ###### att_W ######
        #d_q = f_dim//n_head
        self.w_q = nn.Linear(f_dim, d_q ,bias = False)
        self.w_k = nn.Linear(f_dim, d_q ,bias = False)
        self.w_v = nn.Linear(f_dim, d_v ,bias = False)
        self.drop = nn.Dropout(dropout)

        self.fc = nn.Linear(d_v,f_dim)
        self.fc_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(f_dim)
        self.norm2 = nn.LayerNorm(f_dim)
        self.MHSA = nn.MultiheadAttention(d_v, head_size, dropout= dropout)


    def forward(self,x, c):  


        q = self.w_q(x)
        v = self.w_v(c)
        k = self.w_k(c)

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        attn_output, attn_output_weights  = self.MHSA(q,k,v)
        att = attn_output.transpose(0, 1)
        att = self.fc_drop(self.norm1(att + x))

    
        att1 = self.fc(att)
        att = self.fc_drop(self.norm2(att1+ att))

        return att

class Trans(nn.Module):
    def __init__(self,depth, dim, head, d_q, d_v, dropout_rate=0.2):
        super(Trans, self).__init__()


        layers=[]
        for d in range(depth):
            layers.extend([Residual(
                PreNormDrop(dim, dropout_rate, 
                        MultiHead_self(dim, head, d_q, d_v)
                    )
                ),
                Residual(PreNormDrop(dim, dropout_rate, FeedForward(dim, dim*2, dropout_rate)))])
        self.net = nn.Sequential(*layers)
        self.net_dual = MultiHead_two(dim,head,d_q,d_v)
    def forward(self,x,c):
        return self.net_dual(self.net(x),self.net(c))


class Trans_one(nn.Module):
    def __init__(self,depth, dim, head, d_q, d_v, dropout_rate=0.2):
        super(Trans_one, self).__init__()


        layers=[]
        for d in range(depth):
            layers.extend([Residual(
                PreNormDrop(dim, dropout_rate, 
                        MultiHead_self(dim, head, d_q, d_v)
                    )
                ),
                Residual(PreNormDrop(dim, dropout_rate, FeedForward(dim, dim*2, dropout_rate)))])
        self.net = nn.Sequential(*layers)
        #self.net_dual = MultiHead_two(dim,head,d_q,d_v)
    def forward(self,x):

        return self.net(x)


        



class Trans_for_two(nn.Module):
    def __init__(self,depth, dim, head, d_q, d_v, dropout_rate=0.2):
        super(Trans_for_two, self).__init__()


        self.layers = nn.ModuleList([])
        for d in range(depth):
            self.layers.append(nn.ModuleList([
                Trans_one(1, dim, head, d_q, d_v, dropout_rate=0.2),
                Trans_one(1, dim, head, d_q, d_v, dropout_rate=0.2),
                MultiHead_with_context(dim, head, d_q, d_v),
                MLP([d_q*2,d_q*3//4, d_q*3//4 , d_q]), 
                MLP([d_q*2,d_q*3//4, d_q*3//4 , d_q])
            ]))

        self.depth = depth

        
    def forward(self,x,c):


        for t1,t2, t_both, fc1, fc2 in self.layers:
            x0= x
            c0= c
            x = t1(x)
            c = t2(c)
            x1 = t_both(x,c)
            c1 = t_both(c,x)    # This part can possible be removed

            x = x0 + fc1(torch.cat((x,x1), dim=2))
            c = c0 + fc2(torch.cat((c,c1), dim=2) )         # This as well
            '''
            Add here normalization 
            '''
        return x,c







class Linear_net_small(nn.Module):
    def __init__(self,depth, dim, head, d_q, d_v,kp_in=5,  dropout_rate=0.2 ):
        super(Linear_net_small, self).__init__()
        self.prep_im = MLP([dim,d_q,d_q], do_bias=False)
        self.prep_p = MLP([dim,d_q,d_q], do_bias=False)
        self.pos_enc = MLP([kp_in, 32,64,128,d_q//2, d_q], do_bias=False)
        ########################## Trans Part ##################
        self.tras_im = Trans_for_two(depth, d_q, head, d_q,d_q )
        self.self_att_points = Trans_one(6, d_q, head, d_q, d_v, dropout_rate=dropout_rate)
        
        def_cofig = {
            "d_model": d_q,
            "nhead": head,
            "attention":"linear",
            "layer_names": ['self', 'cross']*depth

        }

        self.poins_plus_im =LocalFeatureTransformer(def_cofig) 

        #### change after, that is for temp optima ltransp
        self.w_q = nn.Linear(d_q, d_q ,bias = False)
        self.w_k = nn.Linear(d_q, d_q ,bias = False)
        self.w_v = nn.Linear(d_q, d_v ,bias = False)
        self.MHSA = nn.MultiheadAttention(d_v, 1, dropout= dropout_rate)
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.occl = nn.Parameter(torch.zeros(1, 1, d_q))
        self.d_q = d_q



    def forward(self,im1,im2,poitns, im_points, points_pos, ocl = False):
       
        #### Pos encoding ####
        im2 = im2 + self.pos_enc(im_points)
        poitns = poitns + self.pos_enc(points_pos)
        b = im2.shape[0]
        if(ocl):
            cls_occl = self.occl.expand(b,-1,-1)
            im2 = torch.cat((im2, cls_occl), dim =1)


        p, im =  self.poins_plus_im(poitns, im2)
        im = im.transpose(1,2)
        scores = torch.matmul(p,im)/p.shape[2]**.5
        scores = nn.functional.softmax(scores, 2)


        return p, scores


# !/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Copyright 2025 Baidu Inc. All Rights Reserved.
2024/3/6, by SJGLAB@163.com, create

"""

from __future__ import print_function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import math
import numpy as np
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_sequence
from Bio.Seq import Seq
from typing import Tuple
import itertools
from collections import defaultdict

class XCA(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, gpu=True, attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1,device=torch.device('cuda' if gpu else 'cpu')))
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gpu = gpu
        if self.gpu:
            self.temperature = self.temperature.cuda()
            self.qkv = self.qkv.cuda()
            self.attn_drop = self.attn_drop.cuda()
            self.proj = self.proj.cuda()
            self.proj_drop = self.proj_drop.cuda()
        
    def forward(self,x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,4,1)
        q,k,v = qkv.unbind(0)
        
        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)
        attn = (q@k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn@v).permute(0,3,1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class XCA_label(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, gpu=True, attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1,device=torch.device('cuda' if gpu else 'cpu')))
        self.qkv = nn.Linear(dim,dim*2,bias=qkv_bias)
        # self.v = nn.Linear(65,dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gpu = gpu
        if self.gpu:
            self.temperature = self.temperature.cuda()
            self.qkv = self.qkv.cuda()
            self.attn_drop = self.attn_drop.cuda()
            self.proj = self.proj.cuda()
            self.proj_drop = self.proj_drop.cuda()
        
    def forward(self,x,y): 
        B,N,C = x.shape
        B_1,N_1,C_1 = y.shape
        qkv = self.qkv(x).reshape(B,N,2,self.num_heads,C//self.num_heads).permute(2,0,3,4,1)
        v = y.reshape(B_1,N_1,self.num_heads,C_1//self.num_heads).permute(0,2,3,1)
        q,k = qkv.unbind(0)
        
        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)
        attn = (q@k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn@v)

        x = x.permute(0,3,1,2).reshape(B_1,N_1,C_1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class XCA_label_1(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, gpu=True, attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1,device=torch.device('cuda' if gpu else 'cpu')))
        self.kv = nn.Linear(dim,dim*2,bias=qkv_bias)
        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gpu = gpu
        if self.gpu:
            self.temperature = self.temperature.cuda()
            self.kv = self.kv.cuda()
            self.q = self.q.cuda()
            self.attn_drop = self.attn_drop.cuda()
            self.proj = self.proj.cuda()
            self.proj_drop = self.proj_drop.cuda()
        
    def forward(self,x,y): 
        #x:lstm emb
        #y:label emb
        B,N,C = x.shape
        B_1,N_1,C_1 = y.shape
        q = self.q(x).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        kv = self.kv(y).reshape(B_1,N_1,2,self.num_heads,C_1//self.num_heads).permute(2,0,3,1,4)

        k,v = kv.unbind(0)
        
        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)
        attn = (q@k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn@v)
        # print(x.shape)
        x = x.permute(0,2,1,3).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
                 drop=0., kernel_size=3):
        super().__init__()

        out_features = out_features or in_features

        padding = kernel_size//2

        self.conv1 = torch.nn.Conv1d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        self.bn = nn.SyncBatchNorm(in_features)
        self.conv2 = torch.nn.Conv1d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x):
        # B, N, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)

        return x

class attention_xca_label_stack(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, data):
        
        super(attention_xca_label_stack, self).__init__()
        self.gpu = data.HP_gpu
                
        self.xca_attention_1 = XCA_label_1(data.HP_hidden_dim, num_heads=20, qkv_bias=False, gpu=self.gpu)
        self.xca_attention_2 = XCA(data.HP_hidden_dim, num_heads=20, qkv_bias=False, gpu=self.gpu,attn_drop=0.,proj_drop=0.)
        
        self.LPI = LPI(data.HP_hidden_dim,kernel_size=9)
        
        self.ffn = nn.Sequential(nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim),nn.ReLU(),
                                nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim))

        self.norm1 = nn.LayerNorm(data.HP_hidden_dim)
        self.norm2 = nn.LayerNorm(data.HP_hidden_dim)
        

        if self.gpu:
            self.LPI = self.LPI.cuda()
            self.xca_attention_1 = self.xca_attention_1.cuda()
            self.xca_attention_2 = self.xca_attention_2.cuda()
            
            self.norm1 = self.norm1.cuda()
            self.norm2 = self.norm2.cuda()
            self.ffn = self.ffn.cuda()
            
            

    def forward(self, out, label_embs):
        out = self.xca_attention_1(out,label_embs) + out
        out = self.xca_attention_2(out)+out
        out = self.norm1(out)
                  
        out = self.LPI(out)  + out
        out = self.norm2(out)
        
        out = self.ffn(out) + out
        return out
    
class attention_xca_stack(nn.Module):

    def __init__(self, data):
        
        super(attention_xca_stack, self).__init__()
        self.gpu = data.HP_gpu
                
        self.xca_attention = XCA(data.HP_hidden_dim, num_heads=20, qkv_bias=False, gpu=self.gpu)
        
        self.LPI = LPI(data.HP_hidden_dim,kernel_size=9)
        
        self.norm1 = nn.LayerNorm(data.HP_hidden_dim)
        self.norm2 = nn.LayerNorm(data.HP_hidden_dim)
        
        self.ffn = nn.Sequential(nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim),nn.ReLU(),
                                nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim))

        if self.gpu:
            self.xca_attention = self.xca_attention.cuda()
            self.LPI = self.LPI.cuda()            
            self.norm1 = self.norm1.cuda()
            self.norm2 = self.norm2.cuda()
            self.ffn = self.ffn.cuda()
            
            
    def forward(self, out):
        
        out = self.xca_attention(out) + out
        out = self.norm1(out)
        
        out = self.LPI(out)  + out
        out = self.norm2(out)
        
        out = self.ffn(out) + out
        return out


# !/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Copyright 2025 Baidu Inc. All Rights Reserved.
2024/3/6, by SJGLAB@163.com, create

"""

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
from .wordrep import WordRep
from model.attention import XCA,XCA_label_1,LPI,attention_xca_stack,attention_xca_label_stack
import math

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..." % (data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.wordrep = WordRep(data)
        self.num_of_lstm_layer = data.N_layer
        
        self.bilstm_1 = nn.LSTM(data.word_emb_dim, data.HP_hidden_dim // 2, num_layers=2, batch_first=True,
                                  bidirectional=True)
        
        self.bilstm_2 = nn.LSTM(data.word_emb_dim, data.HP_hidden_dim // 2, num_layers=2, batch_first=True,
                                  bidirectional=True) 
        
        self.attention_xca_stack = nn.ModuleList(
            [attention_xca_stack(data) for _ in range(self.num_of_lstm_layer)])
        
        
        self.attention_xca_label_stack = nn.ModuleList(
            [attention_xca_label_stack(data) for _ in range(self.num_of_lstm_layer)])
        
        self.ffn = nn.Linear(data.HP_hidden_dim,64)

        if self.gpu:
            self.bilstm_1 = self.bilstm_1.cuda()
            self.bilstm_2 = self.bilstm_2.cuda()
            self.attention_xca_stack = self.attention_xca_stack.cuda()
            self.attention_xca_label_stack = self.attention_xca_label_stack.cuda()
            self.ffn = self.ffn.cuda()


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                input_label_seq_tensor):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
                label_size: nubmer of label
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent, label_embs = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs,
                                                  char_seq_lengths, char_seq_recover, input_label_seq_tensor)
        
        out_1 = word_represent
        out_2 = word_represent
        
        hidden = None
        out_1, hidden = self.bilstm_1(out_1, hidden)
        
        hidden = None
        out_2, hidden = self.bilstm_2(out_2, hidden)
       
        for attention_stack in self.attention_xca_stack:
            out_1 = attention_stack(out_1) + out_1
            
        for attention_xca_label_stack in self.attention_xca_label_stack:
            out_2 = attention_xca_label_stack(out_2,label_embs) + out_2
            
        out = out_1 + out_2
        
        out = self.ffn(out)
                               
        return out
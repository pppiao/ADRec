#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
from operator import truediv
from tokenize import group
from turtle import hideturtle, pos
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.init import *
import pandas as pd
import os
import time
from transformers import AutoTokenizer, AutoModel
torch.set_printoptions(sci_mode=False)
# torch.autograd.set_detect_anomaly(True)


class MocoQueue(nn.Module):
    def __init__(self, encoder_q = None, encoder_k = None, opt = None, M=0.999):      # base=768  large=1024
        super(MocoQueue, self).__init__()
        """
            dim: feature dimension (default: 768)
            K: queue size; number of negative keys (default: 2.5*batchsize, From ESimCSE)
            m: moco momentum of updating key encoder (default: 0.999)
            T: softmax temperature (default: 0.05)
        """
        self.queue_size = int(opt.queueSize)
        self.M = M
        stdv = 1.0 / math.sqrt(opt.hiddenSize)
        init_embedding_sess = torch.randn(self.queue_size, opt.hiddenSize).uniform_(-stdv, stdv)
        init_embedding_target = torch.randn(self.queue_size, opt.hiddenSize).uniform_(-stdv, stdv)
        self.register_buffer("queue_sess", init_embedding_sess / (torch.norm(init_embedding_sess, dim=-1).unsqueeze(-1) + 1e-6) )
        self.register_buffer("queue_target", init_embedding_target / (torch.norm(init_embedding_target, dim=-1).unsqueeze(-1) + 1e-6) )
        # print(self.queue)
        # self.reset_parameters() # 只随机初始化queue
        # print(self.queue)

        self.encoder_q = encoder_q          # 浅拷贝,应该会一直指向原始模型数据
        self.encoder_k = encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue_sess_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_target_ptr", torch.zeros(1, dtype=torch.long))

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.M + param_q.data * (1. - self.M)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, type):
        """
            https://github.com/microsoft/unilm/blob/master/infoxlm/src-infoxlm/infoxlm/models/infoxlm.py#L78
        """
        if type == "sess":
            queue_ptr = self.queue_sess_ptr
            queue = self.queue_sess
        elif type == "target":
            queue_ptr = self.queue_target_ptr
            queue = self.queue_target

        # batch_size = keys.size(0)
        # ptr = int(queue_ptr)
        # # 计算实际可以插入的 keys 数量，以避免超出队列大小
        # actual_batch_size = min(batch_size, self.queue_size)

        # # 确保不会因为指针位置导致插入数据时超出队列范围
        # if ptr + actual_batch_size > self.queue_size:
        #     remaining_space = self.queue_size - ptr
        #     # 分两部分处理：先填满队列尾部
        #     queue[ptr:, :] = keys[:remaining_space, :]
        #     # 如果还有剩余，则从队列头开始
        #     if actual_batch_size > remaining_space:
        #         overflow = actual_batch_size - remaining_space
        #         queue[:overflow, :] = keys[remaining_space:actual_batch_size, :]
        #     ptr = overflow
        # else:
        #     # 如果没有超范围，则直接插入
        #     queue[ptr:ptr+actual_batch_size, :] = keys[:actual_batch_size, :]
        #     ptr += actual_batch_size

        ##############################################

        batch_size = keys.size(0)
        ptr = int(queue_ptr)
        # assert self.queue_size % batch_size == 0

        if batch_size >= self.queue_size:
            queue[0:self.queue_size] = keys[0:self.queue_size]
        else:
            if ptr + batch_size <= self.queue_size:
                queue[ptr:ptr+batch_size, :] = keys
                ptr = (ptr + batch_size) % self.queue_size
            else:
                left_len = self.queue_size - ptr
                queue[ptr:, :] = keys[:left_len, :]
                ptr = batch_size-left_len
                queue[:ptr, :] = keys[left_len:, :]
            queue_ptr[0] = ptr




class SessionGraph(Module):
    def __init__(self, opt):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.plm_hiddenSize = opt.plm_hiddenSize
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid

        self.pos_embedding = nn.Embedding(400, self.hidden_size)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(0.1)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4, batch_first = True, dropout = 0, norm_first = True, dim_feedforward = 1024, layer_norm_eps = 1e-4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)

        # self.ln1 = torch.nn.LayerNorm([self.n_node-1,],elementwise_affine=False)

        self.reset_parameters()
        self.bert = AutoModel.from_pretrained(opt.bert_path, output_hidden_states=True)
        self.layer_weights = nn.Parameter(
                torch.tensor([1] * 13, dtype=torch.float, requires_grad=True)
            )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.scaler = torch.cuda.amp.GradScaler()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, data):

        ############################################### 一、转换到cuda上
        his_input_ids, his_attention_mask, label_input_ids, label_attention_mask, mask, sess_id = data

        his_input_ids = trans_to_cuda(his_input_ids)
        his_attention_mask = trans_to_cuda(his_attention_mask)
        label_input_ids = trans_to_cuda(label_input_ids)
        label_attention_mask = trans_to_cuda(label_attention_mask)

        mask = trans_to_cuda(mask)
        sess_id = trans_to_cuda(sess_id)


        ############################################### 二、bert encoding
        with torch.cuda.amp.autocast():
            # with torch.no_grad():
            tmp1 = his_attention_mask.shape[0]
            tmp2 = his_attention_mask.shape[1]
            his_input_ids = his_input_ids.view(-1, his_input_ids.shape[2]) # (batch_size x seq_len) x text_len
            his_attention_mask = his_attention_mask.view(-1, his_attention_mask.shape[2]) # (batch_size x seq_len) x text_len

            his_hidden = self.bert(input_ids=his_input_ids, attention_mask=his_attention_mask).hidden_states # (batch_size x seq_len) x text_len x hidden_size
            label_hidden = self.bert(input_ids=label_input_ids, attention_mask = label_attention_mask).hidden_states # batch_size x text_len x hidden_size

            his_hidden = his_hidden[1]
            label_hidden = label_hidden[1]

            his_hidden = his_hidden.view(tmp1, tmp2, his_hidden.shape[1], -1) # batch_size x seq_len x text_len x hidden_size
            his_attention_mask = his_attention_mask.view(tmp1, tmp2, his_attention_mask.shape[1]) # batch_size x seq_len x text_len
            
            his_hidden = torch.sum(his_hidden, 2) / torch.sum(his_attention_mask, -1).unsqueeze(-1) # batch_size x seq_len x hidden_size
            label_hidden = torch.sum(label_hidden, 1) / torch.sum(label_attention_mask, -1).unsqueeze(-1) # batch_size x hidden_size

            # his_hidden = his_hidden[:,:,7,:]
            # label_hidden = label_hidden[:,7,:]
            

        ############################################### 三、session encoding
        hidden = his_hidden
        with torch.cuda.amp.autocast():
            ############### transformers

            sess_max_len = hidden.shape[1]
            pos_emb = self.pos_embedding.weight[:sess_max_len].unsqueeze(0).repeat(hidden.shape[0],1,1) # batch_size x seq_length x hidden_size
            
            # session_len = torch.sum(mask, -1) # batch_size
            # reversed_pos = session_len.unsqueeze(-1) - 1 - torch.arange(0, sess_max_len).cuda().unsqueeze(0) # batch_size x seq_length
            # reversed_pos = torch.where(reversed_pos>=0, reversed_pos, torch.zeros_like(reversed_pos))
            # reversed_pos_emb = self.pos_embedding(reversed_pos) # batch_size x seq_length x hidden_size

            hidden_with_pos = hidden  # batch_size x seq_length x hidden_size

            mask_transformer = (mask != 1)
            
        transformer_outputs = self.transformer_encoder(src = hidden_with_pos, src_key_padding_mask = mask_transformer) # batch_size x seq_length x hidden_size
        transformer_outputs = transformer_outputs[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x hidden_size
        sess_mean_emb = torch.sum(hidden * mask.unsqueeze(-1), 1) / torch.sum(mask, -1).unsqueeze(-1) # batch_size x hidden_size


        with torch.cuda.amp.autocast():
            session_emb = (transformer_outputs + sess_mean_emb)/2 # batch_size x hidden_size


        return label_hidden, sess_id, session_emb


    def input_target_loss_compute(self, input, target, moco_queue, target_type, is_update_queue):
        
        #### 负样本（从moco队列中取出）
        if target_type == "sess":
            queue_embs = moco_queue.queue_sess.clone().detach() # queue_size x hidden_size
        elif target_type == "target":
            queue_embs = moco_queue.queue_target.clone().detach() # queue_size x hidden_size

        # if target_type == "sess":
        #     ##### 训练时全都加linear 测试时不使用
        #     input = self.linear_1(input)
        #     target = self.linear_1(target)
        #     queue_embs = self.linear_1(queue_embs)
        #     print(self.linear_1.weight)

        ####################### 计算logit
        input_norm = input / (torch.norm(input, dim=-1).unsqueeze(-1) + 1e-6) # batch_size x hidden_size
        #### 正样本
        target_norm =  target / (torch.norm(target, dim=-1).unsqueeze(-1) + 1e-6) # batch_size x hidden_size
        target_norm = target_norm.detach()
        cos_sim_pos = torch.sum(input_norm * target_norm, 1).unsqueeze(-1) # batch_size x 1
        theta_pos = torch.acos(cos_sim_pos) # batch_size x 1
    
        queue_embs_norm = queue_embs / (torch.norm(queue_embs, dim=-1).unsqueeze(-1) + 1e-6) # batch_size x queue_size x hidden_size
        cos_sim_neg = torch.matmul(input_norm, queue_embs_norm.transpose(1, 0)) # batch_size x queue_size

        cos_sim_neg = cos_sim_neg # batch_size x neg_num ==> batch_size x ( (neg_choice_k x (neg_choice_k -1) ) + queue_size)
        theta_neg = torch.acos(cos_sim_neg) # batch_size x queue_size
        
        #### 计算loss
        s = 18
        m = 0.1

        # arc_loss = -1 * (s * torch.cos(theta_pos + m) - torch.log((  torch.exp(s * torch.cos(theta_pos + m))  +   torch.sum(torch.exp(s * torch.cos(theta_neg)), 1).unsqueeze(-1)  )))
        # arc_loss = torch.mean(arc_loss)

        arc_loss = torch.exp(s * torch.cos(theta_pos + m)) / (  torch.exp(s * torch.cos(theta_pos + m))  +   torch.sum(torch.exp(s * torch.cos(theta_neg)), 1).unsqueeze(-1)  )
        arc_loss = -1 * torch.log(arc_loss)
        arc_loss = torch.mean(arc_loss)


        ### 更新队列
        if is_update_queue:
            moco_queue._dequeue_and_enqueue(target, target_type)
        return arc_loss



    def loss_compute(self, label_hidden, session_emb, label_hidden_moco, session_emb_moco, moco_queue):

        ###################首先，更新encoder_k的参数
        moco_queue._momentum_update_key_encoder()

        loss_sess_label = self.input_target_loss_compute(input = session_emb, target = label_hidden_moco, 
                    moco_queue = moco_queue, target_type = "target", is_update_queue = True)
        loss_label_sess = self.input_target_loss_compute(input = label_hidden, target = session_emb_moco, 
                    moco_queue = moco_queue, target_type = "sess", is_update_queue = True)
        
        loss_sess_sess = self.input_target_loss_compute(input = session_emb, target = session_emb_moco, 
                    moco_queue = moco_queue, target_type = "sess", is_update_queue = False)
        loss_label_label = self.input_target_loss_compute(input = label_hidden, target = label_hidden_moco, 
                    moco_queue = moco_queue, target_type = "target", is_update_queue = False)


        arc_loss = 0.3 * loss_sess_label + 0.3 * loss_label_sess  + 0.2 * loss_sess_sess + 0.2 * loss_label_label
        return arc_loss, loss_sess_label, loss_label_sess



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def train_opt(model, data, opt, moco_queue, epoch, data_step):
    model.optimizer.zero_grad()

    if opt.auto_mix_prec:
        label_hidden, _, session_emb = model(data)
        label_hidden_moco, _, session_emb_moco = moco_queue.encoder_k(data)
        label_hidden_moco = label_hidden_moco.detach()
        session_emb_moco = session_emb_moco.detach()
        with torch.cuda.amp.autocast():
            loss, loss_sess_label, loss_label_sess = model.loss_compute(label_hidden, session_emb, label_hidden_moco, session_emb_moco, moco_queue)
        # if epoch == 0 and (data_step  - 2) * opt.batchSize <= opt.queueSize:
        #     return trans_to_cuda(torch.tensor([10])),trans_to_cuda(torch.tensor([10])),trans_to_cuda(torch.tensor([10]))
        model.scaler.scale(
            loss
        ).backward()

        model.scaler.unscale_(model.optimizer)
        model.scaler.step(model.optimizer)
        model.scaler.update()
    else:
        label_hidden, session_emb, label_input_ids = forward(model, data, is_train=True, moco_queue = moco_queue)
        loss = model.loss_compute(label_hidden, session_emb, moco_queue, epoch, label_input_ids)## 这里为什么要用targets - 1呢？是因为foward在计算scores是没有计算mask的
        loss.backward()
        model.optimizer.step()
    
    return loss, loss_sess_label, loss_label_sess

def train(model, train_data, opt, epoch, is_shuffle, is_learn_rate_decay, moco_queue):
    if is_learn_rate_decay:
        print("学习率降低")
        model.scheduler.step()
        print(model.scheduler.get_last_lr()[0])
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    ## slices是装有n_batch个batch的list，每个batch里装有batchsize个索引（用以去取session，但要注意不是session_id），即[[2,7,5],[3,6,9]。。。。]
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=32, batch_size=opt.batchSize, ## , num_workers=4
                                               shuffle=is_shuffle, pin_memory=True)

    train_loader = tqdm(enumerate(train_loader) ,total=train_loader.__len__())
    time_old = time.time()
    
    for data_step, data in train_loader:
        loss, loss_sess_label, loss_label_sess = train_opt(model, data, opt, moco_queue, epoch, data_step)
        total_loss += loss / train_loader.__len__()
        loss = "%.2f" % (loss.detach().cpu().numpy())
        loss_sess_label = "%.2f" % (loss_sess_label.detach().cpu().numpy())
        loss_label_sess = "%.2f" % (loss_label_sess.detach().cpu().numpy())
        train_loader.set_postfix(loss = loss, loss_sess_label = loss_sess_label, loss_label_sess = loss_label_sess)

        if opt.check_step != None:
            if (data_step + epoch * train_data.__len__()) % opt.check_step == 0:
                checkpoint = {"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": model.optimizer.state_dict(),
                            "epoch": epoch}
                try:
                    os.mkdir(f"{opt.save_dir_root}")
                except:
                    pass
                path_checkpoint = f"{opt.save_dir_root}/checkpoint_bc{opt.batchSize}_epoch{epoch}_step{data_step}.pkl"
                torch.save(checkpoint, path_checkpoint)


    print('\tLoss:\t%.3f' % total_loss)
    
    

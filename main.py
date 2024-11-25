#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import pickle
import time
from utils import  Data, load_train_data
from model_transformer_arcface import *   # model_transformer/model_transformer_more_task_comp/model_only_unsimcse
import numpy as np
import pickle
import faulthandler;faulthandler.enable()
import gc
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
time.sleep(3600*2)
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', help='cuda:0/1')
parser.add_argument('--dataset', default='pretrain_data_process', help='dataset name: pretrain_data_process/Instruments_process/Scientific_process')
parser.add_argument('--bert_path', default='../bert_pretrained/bert-base-uncased', help='dataset name: roberta-base/bert-base-uncased/roberta-base/roberta-large/deberta-v3-base/bert-large-uncased/xlm-roberta-base')
parser.add_argument('--text_length', type=int, default=60, help='text max length')
parser.add_argument('--save_dir_root', default='instruments_ckpt', help='save_dir_root')
parser.add_argument('--batchSize', type=int, default=52, help='input batch size')
parser.add_argument('--queueSize', type=int, default=60000, help='input batch size')
parser.add_argument('--is_queue_warm_up', type=bool, default=True, help='input batch size')
parser.add_argument('--check_step', type=int, default=None, help='input batch size')
parser.add_argument('--is_debug', type=bool, default=False, help='input batch size')
parser.add_argument('--is_all_data', type=bool, default=False, help='input batch size')
parser.add_argument('--auto_mix_prec', type=bool, default=True, help='input batch size')
parser.add_argument('--plm_hiddenSize', type=int, default=768, help='hidden state size')
parser.add_argument('--hiddenSize', type=int, default=768, help='hidden state size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.2, help='learning rate ecay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=0, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.device

def main():
    init_seed(22)
    id_to_side_dict = pickle.load(open(f"../datasets/{opt.dataset}/id_to_side_dict.pickle","rb"))
    tokenizer = AutoTokenizer.from_pretrained(opt.bert_path, do_lower_case=True)
    print("tokenizer加载完毕")
    model = trans_to_cuda(SessionGraph(opt))

    start = time.time()
    
    start_epoch = 0
    
    # if True:
    #     path_checkpoint = f"checkpoint_pretrain_512_20000_6e_4/checkpoint_bc512_epoch0_step1000.pkl"
    #     checkpoint = torch.load(path_checkpoint)
    #     model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    #     # model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # start_epoch = checkpoint['epoch'] + 1
    #     # model.scheduler.last_epoch = start_epoch
        
    moco_queue = trans_to_cuda(MocoQueue(encoder_q = model, encoder_k = SessionGraph(opt), opt = opt))
    try:
        print("model used:",path_checkpoint)
        print("************************************************预训练模型加载完毕********************************************************")
    except:
        pass


    for epoch in range(start_epoch, opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        train_data = load_train_data(opt) # 所有国家的数据

        train_data = Data(train_data, id_to_side_dict, tokenizer, opt)
        train(model, train_data, opt, epoch, is_shuffle = True, is_learn_rate_decay = True, moco_queue = moco_queue)

        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": model.optimizer.state_dict(),
                      "epoch": epoch}
        try:
            os.mkdir(f"{opt.save_dir_root}")
        except:
            pass
        path_checkpoint = f"{opt.save_dir_root}/checkpoint_{epoch}_epoch.pkl"
        torch.save(checkpoint, path_checkpoint)
        

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()

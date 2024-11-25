#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random

def load_train_data(opt):
    train_data_click = [[] for i in range(3)]
    for index,file_name in enumerate(["items_seq", "label", "masks"]):
        part = np.load('../datasets/' + opt.dataset + f"/train/" + file_name + f".npy")
        train_data_click[index] = part

    if opt.is_debug:
        for i in range(len(train_data_click)):
            train_data_click[i] = train_data_click[i][:10]
    return train_data_click


# def text_concat(all_side_text, tokenizer):
#     # price = str(all_side_text[1]) + " price" 
#     title = all_side_text['title']
#     brand = all_side_text['brand']
#     category = all_side_text['category']
#     all_side_order = " category: " + category + "[SEP]brand: " + brand + "[SEP]product title: " + title
#     # return title
#     return all_side_order

# def text_concat(all_side_text, tokenizer):
#     # price = str(all_side_text[1]) + " price" 
#     title = all_side_text['title']
#     brand = all_side_text['brand']
#     category = all_side_text['category']
#     all_side_order = [title] + [category] + [brand]
#     # return title
#     return "[SEP]".join(all_side_order)

def text_concat(all_side_text, tokenizer):
    # price = str(all_side_text[1]) + " price" 
    prompt = "Brought it for [mask]"
    title = all_side_text['title']
    brand = all_side_text['brand']
    category = all_side_text['category']
    all_side_order = [prompt] + [category] + [brand] + [title]
    # return title
    return "[SEP]".join(all_side_order)


class Data(Dataset):
    def __init__(self, data, id_to_side_dict, tokenizer, opt):
        self.inputs, self.label, self.masks = data
        self.length = len(self.inputs)
        self.id_to_side_dict = id_to_side_dict
        self.tokenizer = tokenizer
        self.opt = opt
        self.all_len = []

    def __getitem__(self, index):
        inputs, label, masks = self.inputs[index], self.label[index], \
                    self.masks[index]
        
        # 将序列截断到max_len
        last_index = -1
        for index, i in enumerate(inputs):
            if i == 0:
                last_index = index - 1
                break
        if last_index == -1:
            last_index = len(inputs) - 1

        max_len = 20
        inputs = inputs[max(last_index + 1 - max_len, 0):last_index + 1]
        masks = masks[max(last_index + 1 - max_len, 0):last_index + 1]
        
        inputs = np.concatenate([inputs, [0] * (max_len - len(inputs))])
        masks = np.concatenate([masks, [0] * (max_len - len(masks))])
        ##############################

        inputs_to_text = []

        inputs_to_text = [text_concat(self.id_to_side_dict[i], self.tokenizer)  if i != 0 else "None" for i in inputs]
        label_to_text = text_concat(self.id_to_side_dict[label], self.tokenizer)

        processed_input = self.tokenizer(inputs_to_text, padding='max_length', truncation=True, max_length=self.opt.text_length)
        processed_label = self.tokenizer(label_to_text, padding='max_length', truncation=True, max_length=self.opt.text_length)

        his_input_ids = processed_input['input_ids']
        his_attention_mask = processed_input['attention_mask']

        label_input_ids = processed_label['input_ids']
        label_attention_mask = processed_label['attention_mask']

        return [torch.tensor(his_input_ids).long(), torch.tensor(his_attention_mask).long(), torch.tensor(label_input_ids).long(), torch.tensor(label_attention_mask).long(), torch.tensor(masks).long(), torch.tensor(label).long()]

    def __len__(self):
        return self.length

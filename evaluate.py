
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchtext import data
from torchtext import datasets

from sst_sent import SST_SENT
from sequential_models import RNN_s
from sequential_models import RNN_encoder
from utils import load_data

import matplotlib.pyplot as plt
import numpy as np
import json
from IPython import embed
import os
import glob


root_path = '/home/vasilis/Documents/pytorch_ex/pt_ex1_6march/runs/'
snapshot_path = 'Mar12_10-04-43_vasilis-MS-7A33/best_dev_model/_devacc_0.803030303030303_devloss_0.8549262285232544__iter_100_model.pt'
_filepath = root_path + snapshot_path
gpu_id = 0
model = torch.load(
    _filepath,
    map_location=lambda storage, loc: storage.cuda(gpu_id))
model.eval()

# loading dataset in order to get vocabulary
train, dev, test, inputs, answers = load_data('CSV_FILE', 'CSV_FILE')

input_vocab = inputs.vocab


while True:
    input_var = input("Enter a sentence: ")
    if len(input_var) == 0:
        continue

    # translate sentence to vocabulary ids
    senttnum = [inputs.vocab.stoi[ind] for ind in input_var.split()]

    # reshape to be [batch, size] and load in a tensor variable
    x = torch.from_numpy(np.array(senttnum).reshape((1, len(senttnum))))
    x = Variable(x.type(torch.cuda.LongTensor), requires_grad=False)
    len_x = len(senttnum)

    len_x_t = torch.from_numpy(np.array([len_x]))
    len_x_t = Variable(len_x_t.type(
        torch.cuda.LongTensor), requires_grad=False)

    l_probs, h_l, attention_weights = model(x, len_x_t.data)

    _, predictions = torch.max(l_probs.data, 1)
    verdict = answers.vocab.itos[sum(predictions + 1)]
    print ('\nyour sentence: {} is of class {}\n'.format(input_var, verdict))
    print ("weight distibution {}".format(attention_weights))
    k_ = 5
    topk_classes = torch.topk(l_probs.data, k_)
    topk_classes_list = topk_classes[1].tolist()[0]
    print ("the top {} classes where: ".format(k_))

    for c_i in topk_classes_list:
        class_i = answers.vocab.itos[c_i + 1]
        print ("{} ".format(class_i))

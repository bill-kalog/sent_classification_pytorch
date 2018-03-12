
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils import load_data
import numpy as np
from IPython import embed
import os
import glob

root_path = '/home/vasilis/Documents/pytorch_ex/pt_ex1_6march/runs/'
# specify the folder name
folder_ = 'Mar12_17-58-03_vasilis-MS-7A33'
snapshot_path = os.path.join(root_path, folder_)
# retrieve the path to the saved model
_filepath = glob.glob(os.path.join(snapshot_path, 'best_dev_model', '*'))[0]

gpu_id = 0
model = torch.load(
    _filepath,
    map_location=lambda storage, loc: storage.cuda(gpu_id))
model.eval()

# loading dataset in order to get vocabulary
# TODO somehow should only load the saved fields and not the whole dataset
train, dev, test, inputs, answers = load_data('CSV_FILE', 'CSV_FILE')
input_vocab = inputs.vocab


while True:
    input_var = input("Enter a sentence: ")
    if len(input_var) == 0:
        continue

    # translate sentence to vocabulary ids
    prepr_sent = inputs.preprocess(input_var)
    senttnum = [inputs.vocab.stoi[ind] for ind in prepr_sent]

    # reshape to be [batch, size] and load in a tensor variable
    x = torch.from_numpy(np.array(senttnum).reshape((1, len(senttnum))))
    x = Variable(x.type(torch.cuda.LongTensor), requires_grad=False)
    len_x = len(senttnum)

    len_x_t = torch.from_numpy(np.array([len_x]))
    len_x_t = Variable(len_x_t.type(
        torch.cuda.LongTensor), requires_grad=False)

    l_probs, h_l, attention_weights = model(x, len_x_t.data)
    # get distributions
    prob_distr = F.softmax(l_probs, dim=1)

    _, predictions = torch.max(l_probs.data, 1)
    _, predictions_soft = torch.max(prob_distr.data, 1)
    verdict = answers.vocab.itos[sum(predictions + 1)]
    print ("\nyour sentence: '{}' is of class: {}\n".format(
        prepr_sent, verdict))
    print ("weight distribution {}".format(attention_weights))
    k_ = 5
    topk_classes = torch.topk(prob_distr.data, k_)
    topk_classes_list = topk_classes[1].tolist()[0]
    topk_prob_list = topk_classes[0].tolist()[0]
    print ("the top {} classes where: ".format(k_))

    for idx, c_i in enumerate(topk_classes_list):
        class_i = answers.vocab.itos[c_i + 1]
        print ("{}, prob : {} ".format(class_i, topk_prob_list[idx]))

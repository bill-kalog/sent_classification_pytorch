import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

from sst_sent import SST_SENT
from sequential_models import RNN_s
from sequential_models import RNN_encoder
from convolutional_models import CNN_encoder
import transformer_models
from transformer_models import NoamOpt
from utils import load_data
from conf import config

import matplotlib.pyplot as plt
import numpy as np
import json
from IPython import embed
import os
import glob


# library for using tensorboard with pytorch check
# https://github.com/lanpa/tensorboard-pytorch
# https://medium.com/@dexterhuang/tensorboard-for-pytorch-201a228533c5
from tensorboardX import SummaryWriter


def perform_forward_pass(
        d_batch, model, loss_function):
    '''
    perform the forward pass of a model  for a given batch of data
    and return its loss and accuracy
    '''
    if config['cnn']:
        l_probs, h_l, attention_weights = model(
            d_batch.text[0], sentences_length=d_batch.text[1])
    elif config['transformer']:
        # TODO
        # model.forward(d_batch.text[0], sentences_length=d_batch.text[1])
        l_probs, h_l, attention_weights = model.forward(d_batch.text[0])
    elif config['attention']:
        l_probs, h_l, attention_weights = model(
            d_batch.text[0], sentences_length=d_batch.text[1])
    else:
        # get log probabilities
        l_probs, h_l = model(
            d_batch.text[0], sentences_length=d_batch.text[1])

    # calculate loss
    loss = loss_function(l_probs, d_batch.label - 1)
    # calculate accuracy
    _, predictions = torch.max(l_probs.data, 1)
    k_ = config['k_']
    topk_classes = torch.topk(l_probs.data, k_)[1] + 1
    filter_ = torch.eq(topk_classes, d_batch.label.data.unsqueeze(1))

    acc = torch.sum(torch.eq(predictions, d_batch.label.data - 1)) \
        / predictions.size()[0]
    acc_k = torch.sum(filter_) / predictions.size()[0]
    return acc, loss, h_l, acc_k


def get_attention_weights(d_batch, model, vocab_input, vocab_output, file_path,
                          step, filename):
    '''
    save attention distributions in a json file
    '''
    attention_dir = os.path.join(file_path, 'attention_weights')
    attention_file = os.path.join(attention_dir, filename + '_' + str(step))
    if not os.path.exists(attention_dir):
        os.makedirs(attention_dir)
    if config['transformer']:
        l_probs, h_l, attention_weights = model.forward(d_batch.text[0])
    else:
        l_probs, h_l, attention_weights = model(
            d_batch.text[0], sentences_length=d_batch.text[1])
    _, predictions = torch.max(l_probs.data, 1)
    dic_ = {}
    for sentence_counter in range(len(d_batch.text[0])):
        # gather all information we want to store
        # sentence_in_numbers = d_batch.text[0][sentence_counter]
        sentence_length = d_batch.text[1][sentence_counter]
        prob_distribution = attention_weights[sentence_counter, :, :].data.tolist()[0]
        pred_label = predictions[sentence_counter]
        actual_label = d_batch.label[sentence_counter] - 1
        # get text from integers
        int2text = [vocab_input.itos[int(ind)] for ind in list(
            d_batch.text[0][sentence_counter])]
        pred_label2text = vocab_output.itos[pred_label + 1]
        actual_label2text = vocab_output.itos[int(actual_label) + 1]

        temp = []
        for index_ in range(len(int2text)):
            temp.append((int2text[index_], prob_distribution[index_]))

        dic_['sent_id_' + str(sentence_counter)] = {
            "mass": sum(prob_distribution[:sentence_length]),
            "sentence_length": sentence_length,
            "pred_label": pred_label2text,
            "actual_label": actual_label2text,
            "sentence": temp

        }

        # # uncomment to have distr and sentence in different lists
        # dic_['sent_id_' + str(sentence_counter)] = {
        #     "mass": sum(prob_distribution[:sentence_length]),
        #     "sentence_length": sentence_length,
        #     "pred_label": pred_label2text,
        #     "actual_label": actual_label2text,
        #     "dist": prob_distribution,
        #     "text": int2text

        # }
    json.dump(dic_, open(attention_file + ".json", 'w'), indent="\t")
    print("Saved attention weights file at: {}".format(
        attention_file + ".json"))


def train_step(d_batch, model, optimizer, loss_function):
    '''
    Do a single train step
     TODO
    '''
    optimizer.zero_grad()
    outputs, h_n = model(d_batch.text[0], sentences_length=d_batch.text[1])
    # loss = loss_function(l_probs, d_batch.label - 1)
    return


writer = SummaryWriter()
writer_path = list(writer.all_writers.keys())[0]
best_model_path = os.path.join(writer_path, 'best_dev_model')
if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

# pick which dataset to load
train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_SENT')
# train, dev, test, inputs, answers = load_data('CSV_FILE', 'CSV_FILE')

train_iter, dev_iter2, test_iter3 = data.BucketIterator.splits(
    (train, dev, test),
    batch_sizes=(100, len(dev.examples), len(test.examples)),
    sort_key=lambda x: len(x.text), device=0, sort_within_batch=True)


train_iter.init_epoch()
dev_iter2.init_epoch()
test_iter3.init_epoch()


# define a model
# dnn_model = RNN_s(
#     input_dim=inputs.vocab.vectors.size()[1],
#     output_dim=200,
#     num_classes=len(answers.vocab.freqs.keys()),
#     vocab=inputs.vocab)
# dnn_model.cuda(0)

if config['cnn']:
    dnn_model = CNN_encoder(
        input_dim=inputs.vocab.vectors.size()[1],
        output_dim=200,
        num_classes=len(answers.vocab.freqs.keys()),
        vocab=inputs.vocab)
    dnn_model.cuda(0)
    # define loss funtion and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(dnn_model.parameters())
    # warmup = 400
    # optimizer = NoamOpt(inputs.vocab.vectors.size()[1], 1, warmup,
    #                     torch.optim.Adam(dnn_model.parameters(), lr=0,
    #                                      betas=(0.9, 0.98), eps=1e-9))
elif config['transformer']:
    dnn_model = transformer_models.make_model(
        src_vocab=inputs.vocab,
        num_classes=len(answers.vocab.freqs.keys()),
        d_model=inputs.vocab.vectors.size()[1],
        h=5)
    dnn_model.cuda(0)
    # TODO add loss criterion that is just a placeholder
    loss_function = nn.NLLLoss()
    # optimizer = optim.Adam(dnn_model.parameters())
    warmup = 400
    optimizer = NoamOpt(inputs.vocab.vectors.size()[1], 1, warmup,
                        torch.optim.Adam(dnn_model.parameters(), lr=0,
                                         betas=(0.9, 0.98), eps=1e-9))
    # embed()
    pass
else:
    dnn_model = RNN_encoder(
        input_dim=inputs.vocab.vectors.size()[1],
        output_dim=200,
        num_classes=len(answers.vocab.freqs.keys()),
        vocab=inputs.vocab)
    dnn_model.cuda(0)
    # define loss funtion and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(dnn_model.parameters())



# optimizer = optim.SGD(dnn_model.parameters(), lr=0.01, momentum=0.9)

# define list to stores training info
train_losses_list = []
train_acc_list = []
dev_losses_list = []
dev_acc_list = []
max_acc = 0
max_acc_k = 0
acc_step = 0
save_graph_of_model = True
print ('Starting training procedure')
# start training procedure
for batch_idx, batch in enumerate(train_iter):

    # switch to to training mode,zero out gradients
    dnn_model.train()
    optimizer.zero_grad()

    # train_step(batch, dnn_model, optimizer, loss_function)
    # TODO refactor training loop

    # pass training batch
    # acc, loss, _ = perform_forward_pass(batch, dnn_model, loss_function)
    acc, loss, _, acc_k = perform_forward_pass(
        batch, dnn_model, loss_function)

    loss.backward()
    optimizer.step()
    train_acc_list.append(acc)
    train_losses_list.append(float(loss))

    writer.add_scalar('train/Loss', float(loss), batch_idx)
    writer.add_scalar('train/Acc', acc, batch_idx)

    # check dev set accuraccy
    dnn_model.eval()
    dev_batch = next(iter(dev_iter2))
    # acc, loss, _ = perform_forward_pass(dev_batch, dnn_model, loss_function)
    acc, loss, _, acc_k = perform_forward_pass(
        dev_batch, dnn_model, loss_function)

    dev_acc_list.append(acc)
    dev_losses_list.append(float(loss))
    if max_acc < acc:
        max_acc = acc
        max_acc_k = acc_k
        acc_step = batch_idx
        # save a snapshot of the new best model, example:
        # https://github.com/pytorch/examples/blob/master/snli/train.py
        best_model_file_path = os.path.join(
            best_model_path, '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(
                max_acc, float(loss), batch_idx))
        torch.save(dnn_model, best_model_file_path)
        # delete any previous model stored
        print (best_model_file_path)
        for f in glob.glob(os.path.join(best_model_path, '*')):
            if f != best_model_file_path:
                os.remove(f)


        #### test the current best model against our test set ####
        # TODO
        test_batch = next(iter(test_iter3))
        t_acc, t_loss, _, acc_k = perform_forward_pass(
            test_batch, dnn_model, loss_function)
        print ('accuraccy on test set is {} and at topk {}'.format(
            t_acc, acc_k))


    writer.add_scalar('dev/Loss', float(loss), batch_idx)
    writer.add_scalar('dev/Acc', acc, batch_idx)

    # getting tracing of legacy functions not supported error
    # if save_graph_of_model:
    #     writer.add_graph(dnn_model, (dev_batch.text[0], ))
    #     save_graph_of_model = False

    # for code aboout checkpionting check
    # https://github.com/pytorch/examples/blob/master/snli/train.py

    # info and stop criteria
    if train_iter.epoch % 1 == 0:
        print ('epoch {} iteration {} max acc {} at k {} at step {} \n'.format(
            train_iter.epoch, batch_idx, max_acc, max_acc_k, acc_step))
        # save attention weights from dev set
        if config['attention']:
            get_attention_weights(
                dev_batch, dnn_model, inputs.vocab, answers.vocab,
                writer_path, batch_idx, 'dev_set')

    if train_iter.epoch > config['epochs']:
        # save_embeddings to tensorboard
        _, _, h_l_r, _ = perform_forward_pass(
            dev_batch, dnn_model, loss_function)
        # out_embd = torch.cat((h_l_r.data, torch.ones(len(h_l_r), 1)), 1)
        out_embd = h_l_r.data[:, :, None]
        writer.add_embedding(
            h_l_r.data, metadata=dev_batch.label.data, global_step=batch_idx)
        # embed()
        break


writer.close()
# dev_acc_list.index(max(dev_acc_list))
# plot accuraccy
print ('Maximum dev accuracy was {} at step {} {}'.format(
    max(dev_acc_list), acc_step, max_acc))
# plt.plot(train_acc_list)
# plt.plot(dev_acc_list)
# plt.show()

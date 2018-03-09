import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

from sst_sent import SST_SENT
from sequential_models import RNN_s
from sequential_models import RNN_encoder

import matplotlib.pyplot as plt
import numpy as np
import json
from IPython import embed
import os


# library for using tensorboard with pytorch check
# https://github.com/lanpa/tensorboard-pytorch
# https://medium.com/@dexterhuang/tensorboard-for-pytorch-201a228533c5
from tensorboardX import SummaryWriter

config = {'attention': True}

def load_data():
    '''
    load data, split dataset and create vocabulary
    '''
    inputs = data.Field(
        lower=True, include_lengths=True, batch_first=True)
    answers = data.Field(
        sequential=False)

    train, dev, test = SST_SENT.splits(inputs, answers)
    print('Building vocabulary')
    inputs.build_vocab(train, dev, test)
    inputs.vocab.load_vectors('glove.6B.300d')

    answers.build_vocab(train)

    return train, dev, test, inputs, answers


def perform_forward_pass(
        d_batch, model, loss_function):
    '''
    perform the forward pass of a model  for a given batch of data
    and return its loss and accuracy
    '''
    if config['attention']:
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
    acc = sum(predictions == d_batch.label.data - 1) \
        / predictions.size()[0]

    return acc, loss, h_l


def get_attention_weights(d_batch, model, vocab_input, vocab_output, file_path,
                          step, filename):
    '''
    save attention distributions in a json file
    '''
    attention_dir = os.path.join(file_path, 'attention_weights')
    attention_file = os.path.join(attention_dir, filename + '_' + str(step))
    if not os.path.exists(attention_dir):
        os.makedirs(attention_dir)

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


train, dev, test, inputs, answers = load_data()
# create data iterators sort_within_batch=True sort batches in descending
# length order that is needed for nn.pack_padded_sequence
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=100, device=0, sort_within_batch=True)
train_iter.init_epoch()

# that is a silly work around in order to have all the dev set in one batch
dev_iter2, test_iter2 = data.BucketIterator.splits(
    (dev, test), batch_size=len(dev.examples), device=0, sort_within_batch=True)
dev_iter2.init_epoch()

dev_iter3, test_iter3 = data.BucketIterator.splits(
    (dev, test), batch_size=len(test.examples), device=0, sort_within_batch=True)
test_iter3.init_epoch()


# define a model
# gru_model = RNN_s(
#     input_dim=inputs.vocab.vectors.size()[1],
#     output_dim=200,
#     num_classes=len(answers.vocab.freqs.keys()),
#     vocab=inputs.vocab)
# gru_model.cuda(0)


gru_model = RNN_encoder(
    input_dim=inputs.vocab.vectors.size()[1],
    output_dim=200,
    num_classes=len(answers.vocab.freqs.keys()),
    vocab=inputs.vocab)
gru_model.cuda(0)


# define loss funtion and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(gru_model.parameters())
# optimizer = optim.SGD(gru_model.parameters(), lr=0.01, momentum=0.9)

# define list to stores training info
train_losses_list = []
train_acc_list = []
dev_losses_list = []
dev_acc_list = []
max_acc = 0
acc_step = 0
save_graph_of_model = True
print ('Starting training procedure')
# start training procedure
for batch_idx, batch in enumerate(train_iter):

    # switch to to training mode,zero out gradients
    gru_model.train()
    optimizer.zero_grad()

    # train_step(batch, gru_model, optimizer, loss_function)
    # TODO refactor training loop

    # pass training batch
    # acc, loss, _ = perform_forward_pass(batch, gru_model, loss_function)
    acc, loss, _ = perform_forward_pass(
        batch, gru_model, loss_function)

    loss.backward()
    optimizer.step()
    train_acc_list.append(acc)
    train_losses_list.append(float(loss))

    writer.add_scalar('train/Loss', float(loss), batch_idx)
    writer.add_scalar('train/Acc', acc, batch_idx)

    # check dev set accuraccy
    gru_model.eval()
    dev_batch = next(iter(dev_iter2))
    # acc, loss, _ = perform_forward_pass(dev_batch, gru_model, loss_function)
    acc, loss, _ = perform_forward_pass(
        dev_batch, gru_model, loss_function)

    dev_acc_list.append(acc)
    dev_losses_list.append(float(loss))
    if max_acc < acc:
        max_acc = acc
        acc_step = batch_idx
        #### test the current best model against our test set ####
        # TODO
        test_batch = next(iter(test_iter3))
        t_acc, t_loss, _ = perform_forward_pass(
            test_batch, gru_model, loss_function)
        print ('accuraccy on test set is {}'.format(t_acc))


    writer.add_scalar('dev/Loss', float(loss), batch_idx)
    writer.add_scalar('dev/Acc', acc, batch_idx)

    # getting tracing of legacy functions not supported error
    # if save_graph_of_model:
    #     writer.add_graph(gru_model, (dev_batch.text[0], ))
    #     save_graph_of_model = False

    # for code aboout checkpionting check
    # https://github.com/pytorch/examples/blob/master/snli/train.py

    # info and stop criteria
    if train_iter.epoch % 1 == 0:
        print ('epoch {} iteration {} max acc {} at step {} \n'.format(
            train_iter.epoch, batch_idx, max_acc, acc_step))
        # save attention weights from dev set
        if config['attention']:
            get_attention_weights(
                dev_batch, gru_model, inputs.vocab, answers.vocab,
                writer_path, batch_idx, 'dev_set')

    if train_iter.epoch > 13:
        # save sentence vectors
        _, _, h_l_r = perform_forward_pass(
            dev_batch, gru_model, loss_function)
        # out_embd = torch.cat((h_l_r.data, torch.ones(len(h_l_r), 1)), 1)
        out_embd = h_l_r.data[:, :, None]
        writer.add_embedding(
            h_l_r.data, metadata=dev_batch.label.data, global_step=batch_idx)
        # embed()
        break


# save_embeddings to tensorboard



writer.close()
# dev_acc_list.index(max(dev_acc_list))
# plot accuraccy
print ('Maximum dev accuracy was {} at step {} {}'.format(
    max(dev_acc_list), acc_step, max_acc))
# plt.plot(train_acc_list)
# plt.plot(dev_acc_list)
# plt.show()

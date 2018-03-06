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
from IPython import embed


# library for using tensorboard with pytorch check
# https://github.com/lanpa/tensorboard-pytorch
# https://medium.com/@dexterhuang/tensorboard-for-pytorch-201a228533c5
from tensorboardX import SummaryWriter


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

    # get log probabilities
    l_probs, h_l = model(d_batch.text[0], sentences_length=d_batch.text[1])

    # calculate loss
    loss = loss_function(l_probs, d_batch.label - 1)
    # calculate accuracy
    _, predictions = torch.max(l_probs.data, 1)
    acc = sum(predictions == d_batch.label.data - 1) \
        / predictions.size()[0]

    return acc, loss, h_l


writer = SummaryWriter()



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
    if train_iter.epoch > 13:
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

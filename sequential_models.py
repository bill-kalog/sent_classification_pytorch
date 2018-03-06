import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

dtype = torch.cuda.FloatTensor  # comment this to run on CPU


class RNN_s(nn.Module):
    """just a simple sequential model"""
    def __init__(self, input_dim, output_dim, num_classes,
                 vocab=None):
        '''
        define the different submodules that are going to be used
        '''
        super(RNN_s, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = num_classes

        if vocab is not None:
            self.embed = nn.Embedding(len(vocab), input_dim).type(dtype)
            self.embed.weight.data.copy_(vocab.vectors)
        self.rnn = nn.GRU(
            self.input_dim, self.output_dim,
            bidirectional=True)
        # introduce fully connected layer
        if self.rnn.bidirectional:
            self.linear = nn.Linear(
                self.rnn.hidden_size * 2, self.num_classes)
        else:
            self.linear = nn.Linear(
                self.rnn.hidden_size, self.num_classes)

    def forward(self, input_sequence, sentences_length):
        '''
        define forward pass structure
        '''

        word_vectors = self.embed(input_sequence)

        # put sequence length first
        word_vectors_transposed = word_vectors.transpose(1, 0)
        sentences_length = sentences_length.tolist()
        packed_vectors = torch.nn.utils.rnn.pack_padded_sequence(
            word_vectors_transposed, sentences_length)


        output, h_n = self.rnn(packed_vectors)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            output)

        # print(output.size())
        # print (output[:, -1, :].size())
        # print(h_n.size())
        # print (output)
        # print ()
        # output.view(output.size(0)*output.size(1)
        # embed()

        last_hidden_layer = torch.cat((h_n[0], h_n[1]), 1)
        # embed()
        fc1 = self.linear(last_hidden_layer).type(dtype)
        log_softmax = F.log_softmax(fc1, dim=1)
        return log_softmax, last_hidden_layer


class RNN_encoder(nn.Module):
    """docstring for RNN_encoder"""
    def __init__(self, input_dim, output_dim, n_layers=1, dropout=0.5,
                 vocab=None):
        super(RNN_encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embed = nn.Embedding(len(vocab), input_dim).type(dtype)
        self.embed.weight.data.copy_(vocab.vectors)
        self.encoder = nn.GRU(
            self.input_dim, self.output_dim, self.n_layers,
            dropout=self.dropout, bidirectional=True)

    def forward(self, input_sentences, sentences_length, hidden_vectors=None):
        '''
        TODO fix sentence length argument
        '''
        word_vectors = self.embed(input_sentences)
        # put seqeunce length as first dimension
        word_vectors_transposed = word_vectors.transpose(1, 0)
        # Pack a Variable containing padded sequences of variable length.
        packed_vectors = torch.nn.utils.rnn.pack_padded_sequence(
            word_vectors_transposed, sentences_length.tolist())
        embed()
        output, h_n = self.encoder(packed_vectors, hidden_vectors)
        # pad shorter outputs with zeros
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            output)
        return output, h_n


class Attention(object):
    """docstring for Attention"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, hidden_vector, encoder_output):
        pass

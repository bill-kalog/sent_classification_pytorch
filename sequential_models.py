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

        last_hidden_layer = torch.cat((h_n[0], h_n[1]), 1)
        # embed()
        fc1 = self.linear(last_hidden_layer).type(dtype)
        log_softmax = F.log_softmax(fc1, dim=1)
        return log_softmax, last_hidden_layer


class RNN_encoder(nn.Module):
    """an RNN with atention"""
    def __init__(self, input_dim, output_dim, num_classes,
                 n_layers=1, dropout=0.5, vocab=None):
        super(RNN_encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.num_classes = num_classes

        self.embed = nn.Embedding(len(vocab), input_dim).type(dtype)
        self.embed.weight.data.copy_(vocab.vectors)
        self.encoder = nn.GRU(
            self.input_dim, self.output_dim, self.n_layers,
            dropout=self.dropout, bidirectional=True)
        self.calc_attention_values = Attention(self.output_dim * 2)
        self.linear = nn.Linear(self.output_dim * 2, self.num_classes)

    def forward(self, input_sentences, sentences_length, hidden_vectors=None):
        '''
        forward pass
        '''
        word_vectors = self.embed(input_sentences)
        # put seqeunce length as first dimension
        word_vectors_transposed = word_vectors.transpose(1, 0)
        # force weights to exist in a continuous chank of memory
        self.encoder.flatten_parameters()
        # Pack a Variable containing padded sequences of variable length.
        packed_vectors = torch.nn.utils.rnn.pack_padded_sequence(
            word_vectors_transposed, sentences_length.tolist())
        output, h_n = self.encoder(packed_vectors, hidden_vectors)
        # pad shorter outputs with zeros
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            output)
        # get attention weights
        attention_weights = self.calc_attention_values(output)

        # do a weighted sum attention
        output = output.transpose(1, 0)
        attention_weights = attention_weights.transpose(1, 0).unsqueeze(1)
        attented_representations = attention_weights.bmm(output).squeeze(1)

        # pass output through a fc layer
        fc_out = self.linear(attented_representations)
        log_softmax = F.log_softmax(fc_out, dim=1)

        return log_softmax, attented_representations, attention_weights


class Attention(nn.Module):
    """docstring for Attention"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # use a single fc layer to get a score
        self.att_nn = nn.Linear(self.hidden_dim, 1)
        # self.linear2 = nn.Linear(self.hidden_dim, 200)
        # self.linear3 = nn.Linear(200, 1)

    def forward(self, encoder_output):
        # attention scores dim [max_len, batch_size, 1]
        attention_scores = self.att_nn(encoder_output)

        # put one more layer in the calculation
        # relu_1 = F.relu(self.linear2(encoder_output))
        # attention_scores = self.linear3(relu_1)

        # transform to a distribution. dim [max_len, batch_size, 1] -> [len, bsize]
        attention_distribution = F.softmax(attention_scores, 0).squeeze(2)

        return attention_distribution

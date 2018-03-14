import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.autograd import Variable
from sequential_models import Attention

dtype = torch.cuda.FloatTensor
dtype2 = torch.cuda.LongTensor


class CNN_encoder(nn.Module):
    """docstring for CNN_encoder"""
    def __init__(self, input_dim, output_dim, num_classes, n_layers=2,
                 dropout=0.5, vocab=None):
        super(CNN_encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.kernel_size = 3

        self.embed = nn.Embedding(len(vocab), input_dim).type(dtype)
        self.embed.weight.data.copy_(vocab.vectors)
        # get positional encodings for sentences up until that size
        self.max_len = 100
        self.position_embedding = nn.Embedding(self.max_len, input_dim)
        self.num_channels = self.input_dim
        
        self.conv = nn.ModuleList(
            [nn.Conv1d(self.num_channels, self.num_channels, self.kernel_size,
                       padding=self.kernel_size // 2)
                for _ in range(self.n_layers)])

        # weighted sum attention
        self.calc_attention_values = Attention(self.num_channels)
        self.linear = nn.Linear(self.num_channels, self.num_classes)

    def forward(self, input_sentences, sentences_length):
        '''
        forward pass
        '''
        # initialize a variable tensor for the positional embeddings
        # ofcourse there are sentences with to much padding the resulst 
        # are going to be slightly skewed because we are assigning positional
        # embeddings up until max_len(current_batch)
        position_ids = torch.arange(
            0, input_sentences.size()[1]).repeat(
            input_sentences.size()[0]).type(dtype2)
        position_ids.resize_as_(input_sentences.data)
        position_ids = Variable(position_ids)

        # positional_embd = self.position_embedding(position_ids)

        word_vectors = self.embed(input_sentences)

        # input_embd = F.dropout(
        #     positional_embd + word_vectors, self.dropout)
        input_embd = F.dropout(word_vectors, self.dropout, self.training)

        input_embd = input_embd.transpose(1, 2)
        # embed()
        cnn_input = input_embd
        for idx, layer_ in enumerate(self.conv):
            cnn_input = F.tanh(layer_(cnn_input) + cnn_input)

        # bring it in form [seq_l, batch, dim] from [batch, dim, seq_len]
        cnn_input_trans = cnn_input.transpose(1, 2).transpose(1, 0)
        attention_weights = self.calc_attention_values(cnn_input_trans)

        # do a weighted sum attention
        cnn_input_trans = cnn_input_trans.transpose(1, 0)
        attention_weights = attention_weights.transpose(1, 0).unsqueeze(1)
        attented_representations = attention_weights.bmm(cnn_input_trans).squeeze(1)

        # pass cnn_input through a fc layer
        fc_out = self.linear(attented_representations)
        log_softmax = F.log_softmax(fc_out, dim=1)

        return log_softmax, attented_representations, attention_weights


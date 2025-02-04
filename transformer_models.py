import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.autograd import Variable
from sequential_models import Attention
import copy
import math

dtype = torch.cuda.FloatTensor
dtype2 = torch.cuda.LongTensor


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    Here we are actutally using only an encoder though
    """
    def __init__(self, encoder, src_embed, dim, num_classes):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.dim = dim
        self.num_classes = num_classes
        self.fc_class = nn.Linear(self.dim, self.num_classes)
        self.mos_module = MoS(dim, 15, num_classes)

        # weighted sum attention
        self.calc_attention_values = Attention(self.dim)

    def forward(self, src, src_mask=None):
        "Take in and process masked src and target sequences."
        return self.calculateLogits(self.encode(src, src_mask))

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def calculateLogits(self, net_output):
        # embed()
        # just sum all dimesnsions (bag of words) no attention
        # sent_repr = torch.sum(net_output, dim=1)

        # average dimensions
        # sent_repr = torch.sum(net_output, dim=1) / net_output.size()[1] # converges/overfits a bit faster

        # or take just the last step
        sent_repr = net_output[:, -1]  # converges slower

        # or use attention instead but it performs a bit bad and scores a bad too
        # get attention weights [sent_length, batch_size]
        # attention_weights = self.calc_attention_values(
        #     net_output.transpose(0, 1))
        # # back to [batch, 1, length]
        # attention_weights = attention_weights.transpose(1, 0).unsqueeze(1)
        # # attented_representations
        # sent_repr = attention_weights.bmm(net_output).squeeze(1)
        # embed()

        if False:
            # normal softmax
            fc_out = self.fc_class(sent_repr)

            log_softmax = F.log_softmax(fc_out, dim=1)
        else:
            # or check the mixture of softmaxes module
            log_softmax = self.mos_module(sent_repr)

        return log_softmax, sent_repr  #, attention_weights


class MoS(nn.Module):
    """
        implementation of the mixture of softmaxes module MoS
        http://smerity.com/articles/2017/mixture_of_softmaxes.html
    """
    def __init__(self, hidden_dim, k_num, num_classes):
        super(MoS, self).__init__()
        self.k_num = k_num
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.projections = nn.Linear(hidden_dim, hidden_dim * k_num)
        self.calculate_ps = nn.Linear(hidden_dim, k_num)
        self.fc_class = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden_state):
        nl_projections = torch.tanh(self.projections(hidden_state))
        # split into k parts
        parts = torch.chunk(nl_projections, self.k_num, dim=1)
        # calculate p_k
        p_k = self.calculate_ps(hidden_state)
        p_k_dist = F.softmax(p_k, dim=-1).unsqueeze(2)

        # loop over the k different vectors
        results = []
        for part in parts:
            results.append(
                torch.nn.functional.log_softmax(
                    self.fc_class(part), dim=-1))
        result = torch.cat(results, dim=1).view(
            -1, self.k_num, self.num_classes)
        # p_k_dist = torch.autograd.Variable(torch.ones(100, 15, 1) * (1 / 15)).type(torch.cuda.FloatTensor)
        # embed()
        result = torch.log(p_k_dist) + result
        result = logsumexp(result).view(-1, self.num_classes)
        return result


# This is a numerically stable log(sum(exp(x))) operator (from smerity)
def logsumexp(x):
    max_x, _ = torch.max(x, dim=1, keepdim=True)
    part = torch.log(torch.sum(torch.exp(x - max_x), dim=1, keepdim=True))
    return max_x + part


class Encoder(nn.Module):
    """ encoder for the tranformer based on
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, layer, N_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        # embed()
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module "
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k, dim(d_model)==suggested(512)
        '''
        since we are using pretrained word vectors of dim 300. In our case
        we will be using d_model=300 number of heads== 5 and as a result
        the d_k will be 60
        '''
        self.d_k = d_model // h  # dimensionality suggested 64
        self.h = h  # number of heads i.e num attention layers suggested 8
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # embed()
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(len(vocab), d_model).type(dtype)
        # self.embed.weight.data.copy_(vocab.vectors)
        self.d_model = d_model

    def forward(self, x):
        # return self.lut(x) * math.sqrt(self.d_model)
        return self.embed(x) #* math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, num_classes, N=6,
               d_model=512, d_ff=1024, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    # d_ff = 100
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        d_model, num_classes
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0,
                                    betas=(0.9, 0.98), eps=1e-9))

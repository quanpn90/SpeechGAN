from abc import ABC

import torch
from torch.nn import functional as F
from torch import nn


def get_mask_from_lengths(lengths, n_frames_per_step=1):
    max_len = torch.max(lengths).item()

    if max_len % n_frames_per_step != 0:
        max_len += n_frames_per_step - max_len % n_frames_per_step

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear',
                 factorized=False, n_factors=1, rank=1):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.factorized = factorized
        self.n_factors = n_factors
        self.rank = rank

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

        if self.factorized:
            self.r = torch.nn.Parameter(torch.Tensor(n_factors, rank, out_dim))
            self.s = torch.nn.Parameter(torch.Tensor(n_factors, rank, in_dim))

            self.rm = torch.nn.Parameter(torch.Tensor(n_factors, 1, out_dim))
            self.sm = torch.nn.Parameter(torch.Tensor(n_factors, 1, in_dim ))

            torch.nn.init.constant_(self.rm, 1.0)
            torch.nn.init.constant_(self.sm, 1.0)
            torch.nn.init.normal_(self.r, 0.0, 0.02)
            torch.nn.init.normal_(self.s, 0.0, 0.02)

    def forward(self, x, idx=None):

        if self.factorized:

            assert idx is not None
            r = torch.index_select(self.r, 0, idx).squeeze(0)
            s = torch.index_select(self.s, 0, idx).squeeze(0)
            rm = torch.index_select(self.rm, 0, idx).squeeze(0)
            sm = torch.index_select(self.sm, 0, idx).squeeze(0)

            # weight_ = self.linear_layer.weight * torch.sum(torch.bmm(rm.unsqueeze(-1), sm.unsqueeze(1)), dim=0)
            weight_ = self.linear_layer.weight

            weight_mask = torch.bmm(r.unsqueeze(-1), s.unsqueeze(1))
            weight_mask = torch.sum(weight_mask, dim=0)
            weight_ = weight_ + weight_mask

            return F.linear(x, weight_, self.linear_layer.bias)

        else:
            return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

        # at the moment multilingual factorization doesn't support conv yet
        # possibly we can do
        # weight = [out, in, k] -> reshape [out, in * k] -> factorize into [out], [in * k]

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LocationLayer(torch.nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim,
                 factorized=False, rank=1, n_factors=1):
        """
        :param attention_n_filters: 
        :param attention_kernel_size: 
        :param attention_dim: 
        """
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh',
                                         factorized=factorized, rank=rank, n_factors=n_factors)

    def forward(self, attention_weights_cat, tgt_lang=None):
        # (B, 2, max_time)
        processed_attention = self.location_conv(attention_weights_cat)
        # (B, n_filters, max_time)
        processed_attention = processed_attention.transpose(1, 2)
        # (B, max_time, n_filters)
        processed_attention = self.location_dense(processed_attention, tgt_lang)
        # (B, max_time, attention_dim)
        return processed_attention


class Attention(torch.nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 factorized=False, rank=1, n_factors=1):
        """
        :param attention_rnn_dim:
        :param embedding_dim:
        :param attention_dim:
        :param attention_location_n_filters:
        :param attention_location_kernel_size:
        """
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh',
                                      factorized=factorized, rank=rank, n_factors=n_factors)
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh',
                                       factorized=factorized, rank=rank, n_factors=n_factors)
        self.v = LinearNorm(attention_dim, 1, bias=False,
                            factorized=factorized, rank=rank, n_factors=n_factors)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim, factorized=factorized, rank=rank, n_factors=n_factors)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat, tgt_lang=None):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """
        #  (batch, n_mel_channels * n_frames_per_step)
        processed_query = self.query_layer(query.unsqueeze(1), idx=tgt_lang)  # B, 1,  attn_dim
        processed_attention_weights = self.location_layer(attention_weights_cat, tgt_lang)  # (B, max_time, attn_dim)

        # MLP style attention layer lienar(tanh(query + previous weights * memory))
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory), idx=tgt_lang)

        energies = energies.squeeze(-1)
        return energies  # (B, max_time)

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, tgt_lang=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output #  B,  model_dim
        memory: encoder outputs #  B,  Max_time, model_dim
        processed_memory: processed encoder outputs  #  B, Max_time ,  attn_dim
        attention_weights_cat: previous and cummulative attention weights #  B, 2 ,  Max_time
        mask: binary mask for padded data    #B , Max_time
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat, tgt_lang=tgt_lang)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)  # (B, max_time)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)  # B, 1,model_dim
        attention_context = attention_context.squeeze(1)  # B, model_dim

        return attention_context, attention_weights


class Prenet(torch.nn.Module):
    def __init__(self, in_dim, sizes,
                 factorized=False, rank=1, n_factors=1):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = torch.nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False,
                        factorized=factorized, rank=rank, n_factors=n_factors)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x, tgt_lang=None):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x, tgt_lang)), p=0.5, training=self.training)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x

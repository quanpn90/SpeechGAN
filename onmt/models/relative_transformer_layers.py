import torch
import torch.nn as nn
import onmt

from onmt.models.transformer_layers import PrePostProcessing, MultiHeadAttention, Linear
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.modules.optimized.relative_self_attention import RelativeSelfMultiheadAttn
from onmt.utils import flip
from onmt.modules.bottle import Bottle
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.linear import group_linear, FeedForwardSwish, FeedForward
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout
from onmt.modules.relative_attention import RelPartialLearnableMultiHeadAttn
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.optimized.feed_forward import PositionWiseFeedForward


class RelativeTransformerEncoderLayer(nn.Module):
    # def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False, death_rate=0.0, **kwargs):
    def __init__(self, opt, death_rate=0.0, **kwargs):
        super(RelativeTransformerEncoderLayer, self).__init__()
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)
        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)
        d_head = opt.model_size // opt.n_heads
        if not self.fast_self_attention:
            self.multihead = RelPartialLearnableMultiHeadAttn(opt.n_heads, opt.model_size,
                                                              d_head, dropatt=opt.attn_dropout)
        else:
            self.multihead = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

        if not opt.fast_feed_forward:
            feedforward = FeedForward(opt.model_size, opt.inner_size, opt.dropout, variational=self.variational)
            self.feedforward = Bottle(feedforward)
        else:
            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)

    def forward(self, input, pos_emb, attn_mask, incremental=False, incremental_cache=None, mems=None):

        if incremental and incremental_cache is None:
            incremental_cache = dict()

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:

            # memory for transformer-xl caching
            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            query = self.preprocess_attn(input)
            if not self.fast_self_attention:
                out, _, incremental_cache = self.multihead(query, pos_emb, attn_mask=attn_mask, mems=mems,
                                                           incremental=incremental, incremental_cache=incremental_cache)
            else:
                out, _ = self.multihead(query, pos_emb, attn_mask, None, mems=mems,
                                        incremental=incremental, incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

        if incremental:
            return input, incremental_cache

        return input


class RelativeTransformerDecoderLayer(nn.Module):

    # def __init__(self, h, d_model, p,    d_ff, attn_p=0.1, version=1.0, ignore_source=False,
    #              variational=False, death_rate=0.0):
    def __init__(self, opt, death_rate=0.0):
        super(RelativeTransformerDecoderLayer, self).__init__()
        self.ignore_source = opt.ignore_source
        self.variational = opt.variational_dropout
        self.death_rate = death_rate
        self.fast_self_attention = opt.fast_self_attention
        # self.lfv_multilingual = opt.lfv_multilingual

        self.preprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                  variational=self.variational)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                          variational=self.variational)

            if opt.fast_xattention:
                self.multihead_src = EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)
            else:
                self.multihead_src = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=2)

        self.preprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='n')
        self.postprocess_ffn = PrePostProcessing(opt.model_size, opt.dropout, sequence='da',
                                                 variational=self.variational)

        d_head = opt.model_size // opt.n_heads

        if not self.fast_self_attention:
            self.multihead_tgt = RelPartialLearnableMultiHeadAttn(opt.n_heads, opt.model_size, d_head,
                                                                  dropatt=opt.attn_dropout)
        else:
            self.multihead_tgt = RelativeSelfMultiheadAttn(opt.model_size, opt.n_heads, opt.attn_dropout)

        if not opt.fast_feed_forward:
            feedforward = FeedForward(opt.model_size, opt.inner_size, opt.dropout, variational=self.variational)
            self.feedforward = Bottle(feedforward)
        else:
            self.feedforward = PositionWiseFeedForward(opt.model_size, opt.inner_size, opt.dropout,
                                                       variational=self.variational)

        # if opt.lfv_multilingual:
        #     self.lid_net = lid_net
        #     self.lfv_mapper = nn.Linear(opt.bottleneck_size, opt.model_size)
        # else:
        #     self.lid_net = None
        #     self.lfv_mapper = None

    # def forward(self, input, context, pos_emb, r_w_bias, r_r_bias, mask_tgt, mask_src):
    def forward(self, input, context, pos_emb, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True, mems=None):

        """ Self attention layer
            layernorm > attn > dropout > residual
        """

        if incremental and incremental_cache is None:
            incremental_cache = dict()

        coin = True
        if self.training and self.death_rate > 0:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            # input and context should be time first ?
            if mems is not None and mems.size(0) > 0:
                mems = self.preprocess_attn(mems)
            else:
                mems = None

            query = self.preprocess_attn(input)

            if self.fast_self_attention:
                out, _ = self.multihead_tgt(query, pos_emb, None, mask_tgt, mems=mems,
                                            incremental=incremental, incremental_cache=incremental_cache)
            else:
                out, _, incremental_cache = self.multihead_tgt(query, pos_emb, attn_mask=mask_tgt, mems=mems,
                                                               incremental=incremental,
                                                               incremental_cache=incremental_cache)

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """
            if not self.ignore_source:
                query = self.preprocess_src_attn(input)
                incremental_source = incremental and reuse_source
                out, coverage = self.multihead_src(query, context, context, mask_src,
                                                   incremental=incremental_source,
                                                   incremental_cache=incremental_cache)

                # rescaling before residual
                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                input = self.postprocess_src_attn(out, input)
            else:
                coverage = None

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            # rescaling before residual
            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)
        else:
            coverage = None

        return input, coverage, incremental_cache

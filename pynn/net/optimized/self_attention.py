import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .self_attention_func import self_attn_func
from onmt.constants import double_precision

# from .fast_self_multihead_attn_func          import fast_self_attn_func
# from .fast_self_multihead_attn_norm_add_func import fast_self_attn_norm_add_func
# from apex.normalization.fused_layer_norm     import FusedLayerNorm


if hasattr(torch._C, '_jit_set_profiling_executor'):
    torch._C._jit_set_profiling_executor(False)
if hasattr(torch._C, '_jit_set_profiling_mode'):
    torch._C._jit_set_profiling_mode(False)


@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    # def __init__(self, d_model, n_head, dropout=0.):
    def __init__(self, n_head, d_model, d_k, shared_kv=False, dropout=0.1, norm=True, res=True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_k
        assert self.head_dim * n_head == self.d_model, "d_model must be divisible by n_head"
        self.bias = True
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.out_proj_weight = Parameter(torch.Tensor(d_model, d_model))

        self.in_proj_bias = Parameter(torch.Tensor(3 * d_model))
        self.out_proj_bias = Parameter(torch.Tensor(d_model))
        self.reset_parameters()

        self.attn_func = self_attn_func
        self.optimized = 2
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm = norm
        self.res = res

        try:
            # the fast one requires apex and does not work with incremental so careful
            from apex.contrib.multihead_attn.fast_self_multihead_attn_func import fast_self_attn_func
            self.attn_func_fast = fast_self_attn_func
            self.optimized = 1
        except ModuleNotFoundError as e:
            # print(e)
            # print("Cannot use fast self-attention implementation")
            self.optimized = 2
            self.attn_func_fast = None

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.d_model + self.d_model))
        nn.init.normal_(self.in_proj_weight, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)

        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj_bias, 0.)

    def forward(self, q, k=None, mask=None, scale=1.0, mask_type="key_padding"):
        """
        :param q: input  TxBxH
        :param k: should be None
        :param mask: None or 2D
        :param scale:
        :param mask_type: "key_padding" or time_mask
        :return:
        """
        is_training = self.training
        residual = q if self.res else None
        q = self.layer_norm(q) if self.norm else q

        len_key = q.size(0)
        bsz = q.size(1)
        input_weights = self.in_proj_weight

        input_bias = self.in_proj_bias

        if mask is not None:
            assert mask.dim() == 2, "only accepting two dimensional masks"

        if mask_type == 'key_padding':
            causal_masking = False
        else:  # mask_type == 'causal':
            causal_masking = True

        if self.optimized == 1 and self.training and len_key <= 1024 and q.is_cuda:
            if mask is not None:
                mask = mask.byte()
            output = self.attn_func_fast(causal_masking, is_training, self.n_head, q,
                                         input_weights, self.out_proj_weight,
                                         input_bias, self.out_proj_bias, mask,
                                         False, self.dropout)

            coverage = q.new_zeros((bsz, self.n_head, len_key, len_key))

        else:
            if mask is None:
                mask = mask.bool()
            output, coverage = self.attn_func(causal_masking, is_training, self.n_head, q,
                                               input_weights, self.out_proj_weight,
                                               input_bias, self.out_proj_bias,
                                               mask, self.dropout, False, None)

        output = output * scale

        coverage = coverage.view(bsz, self.n_head, len_key, len_key)
        output = output if residual is None else (output + residual)

        return output, coverage


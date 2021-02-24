import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from .encdec_attention_func import encdec_attn_func

if hasattr(torch._C, '_jit_set_profiling_executor'):
    torch._C._jit_set_profiling_executor(False)
if hasattr(torch._C, '_jit_set_profiling_mode'):
    torch._C._jit_set_profiling_mode(False)


class EncdecMultiheadAttn(nn.Module):
    """Multi-headed encoder-decoder attention.
    See "Attention Is All You Need" for more details.
    """

    # def __init__(self, n_head, d_model, dropout=0.):
    def __init__(self, n_head, d_model, d_k, shared_kv=False, dropout=0.1, norm=True, res=True):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_k

        assert self.head_dim * n_head == self.d_model, "d_model must be divisible by n_head"
        self.bias = False
        self.scaling = self.head_dim ** -0.5  # this value is hardcoded in the "fast" implementation

        self.in_proj_weight_q = Parameter(torch.Tensor(d_model, d_model))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * d_model, d_model))
        self.out_proj_weight = Parameter(torch.Tensor(d_model, d_model))

        self.register_parameter('in_proj_bias_q', None)
        self.register_parameter('in_proj_bias_kv', None)
        self.in_proj_bias_q = None
        self.in_proj_bias_kv = None
        self.out_proj_bias = None

        self.attn_func = encdec_attn_func
        self.layer_norm = nn.LayerNorm(d_model)
        self.norm = norm
        self.res = res

        self.reset_parameters()
        try:
            # the fast one requires apex and does not work with incremental so careful
            from apex.contrib.multihead_attn.fast_encdec_multihead_attn_func import fast_encdec_attn_func
            self.attn_func_fast = fast_encdec_attn_func
            self.optimized = 1

        except ModuleNotFoundError as e:
            # print(e)
            # print("Cannot use fast self-attention implementation")
            self.optimized = 2
            self.attn_func_fast = None

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.in_proj_weight_q)
        # in_proj_weight_kv has shape [2 * hidden, hidden] but it should be
        # initialized like a [hidden, hidden] matrix.
        # sqrt(6 / (hidden + hidden)) / sqrt(6 / (2 * hidden + hidden)) = sqrt(1.5)
        # therefore xavier_uniform gain should be set to sqrt(1.5).
        # nn.init.xavier_uniform_(self.in_proj_weight_kv, gain=math.sqrt(1.5))
        # nn.init.xavier_uniform_(self.out_proj_weight)
        std_ = math.sqrt(2.0 / (self.d_model + self.d_model))
        nn.init.normal_(self.in_proj_weight_q, 0.0, std_)
        nn.init.normal_(self.in_proj_weight_kv, 0.0, std_)
        nn.init.normal_(self.out_proj_weight, 0.0, std_)

    def forward(self, q, k=None, mask=None, scale=1.0):

        assert k is not None
        residual = q if self.res else None
        q = self.layer_norm(q) if self.norm else q
        len_q, bsz, len_k = q.size(0), q.size(1), k.size(0)

        is_training = self.training
        time_masking = False

        if mask is not None:
            assert mask.dim() == 2, "only accepting two dimensional masks"

        if self.optimized == 1 and self.training and len_k <= 1024 and q.is_cuda:
            if mask is not None:
                mask = mask.byte()
            output = self.attn_func_fast(time_masking, is_training, self.n_head, q, k,
                                          self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight,
                                          mask, self.dropout)

            coverage = q.new_zeros((bsz, self.n_head, len_q, len_k))
        # during evaluation we use the python binding which is safer ....
        else:
            output, coverage, = self.attn_func(time_masking, is_training,
                                                self.n_head, q, k,
                                                self.in_proj_weight_q, self.in_proj_weight_kv,
                                                self.out_proj_weight, mask, self.dropout, False, None)

        output = output * scale

        coverage = coverage.view(bsz, self.n_head, len_q, len_k)
        output = output if residual is None else (output+residual)

        return output, coverage


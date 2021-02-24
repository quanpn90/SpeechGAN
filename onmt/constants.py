import torch

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

checkpointing = 0
static = False
residual_type = 'regular'
max_position_length = 8192
torch_version = float(torch.__version__[:3])
double_precision = False

neg_log_sigma1 = 0
neg_log_sigma2 = 4
prior_pi = 0.5


# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import torch.nn as nn
import torch.nn.functional as F

class Ensemble(nn.Module):
    def __init__(self, models):

        super().__init__()
        self.models = models

    def forward(self, src_seq, src_mask, tgt_seq):
        pass
        
    def encode(self, src_seq, src_mask):
        enc_out = []
        mask_out = []
        for model in self.models:
            output, mask = model.encode(src_seq, src_mask)[:2]
            enc_out.append(output)
            mask_out.append(mask)

        return enc_out, mask_out, None

    def decode(self, enc_out, src_mask, tgt_seq):
        dec_out = 0.
        for model, enc, mask in zip(self.models, enc_out, src_mask):
            dec_out += model.decode(enc, mask, tgt_seq)[0]
        dec_out = dec_out / len(self.models)

        return dec_out, None

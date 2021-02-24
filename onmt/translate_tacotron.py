#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import sys
import h5py as h5
import numpy as np
import apex
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.inference.fast_translator import FastTranslator
from onmt.inference.stream_translator import StreamTranslator

parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')

parser.add_argument('-fp16', action='store_true',
                    help='To use floating point 16 in decoding')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-batch_size', type=int, default=1,
                    help='Batch size')

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    if opt.output == "stdout":
        outF = sys.stdout
    else:
        outF = open(opt.output, 'w')

    model = opt.model
    checkpoint = torch.load(model,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']

    model = build_model(model_opt, checkpoint['dicts'])
    optimize_model(model)
    model.load_state_dict(checkpoint['model'])

    if opt.fp16:
        model = model.half()

    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    model.eval()
    if opt.encoder_type == "audio" and opt.asr_format == "scp":
        import kaldiio
        from kaldiio import ReadHelper
        audio_data = iter(ReadHelper('scp:' + opt.src))

    in_file = None
    src_batch = []
    if opt.encoder_type == "audio":
        i = 0
        while True:
            if opt.asr_format == "h5":
                if i == len(in_file):
                    break
                line = np.array(in_file[str(i)])
                i += 1
            elif opt.asr_format == "scp":
                try:
                    name, line = next(audio_data)
                except StopIteration:
                    break

        if opt.stride != 1:
            line = line[0::opt.stride]
        line = torch.from_numpy(line)
        if opt.concat != 1:
            add = (opt.concat - line.size()[0] % opt.concat) % opt.concat
            z = torch.FloatTensor(add, line.size()[1]).zero_()
            line = torch.cat((line, z), 0)
            line = line.reshape((line.size()[0] // opt.concat, line.size()[1] * opt.concat))



        output = model.inference(line)
        # wave_glow to save wav file
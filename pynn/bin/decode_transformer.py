#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import os
import copy
import random
import numpy as np
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pynn.decoder.seq2seq import beam_search
from pynn.util import write_ctm, write_stm, write_text
from pynn.io.kaldi_seq import ScpStreamReader
from pynn.net.tf import Transformer
 
parser = argparse.ArgumentParser(description='pynn')

parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--lm-dic', help='language model dictionary', default=None)
parser.add_argument('--lm-scale', help='language model scale', type=float, default=0.5)

parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--word-dict', help='word dictionary file', default=None)
parser.add_argument('--data-scp', help='path to data scp', required=True)
parser.add_argument('--downsample', help='concated frames', type=int, default=4)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')

parser.add_argument('--batch-size', help='batch size', type=int, default=32)
parser.add_argument('--beam-size', help='beam size', type=int, default=10)
parser.add_argument('--max-len', help='max len', type=int, default=400)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--coverage', help='coverage term', type=float, default=0)
parser.add_argument('--len-norm', help='length normalization', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='hypos/H_1_LV.ctm')
parser.add_argument('--format', help='output format', type=str, default='ctm')
parser.add_argument('--space', help='space token', type=str, default='<space>')

if __name__ == '__main__':
    args = parser.parse_args()

    dic = None
    if args.dict is not None:
        dic = {}
        fin = open(args.dict, 'r')
        for line in fin:
            tokens = line.split()
            dic[int(tokens[1])] = tokens[0]
    word_dic = None
    if args.word_dict is not None:
        fin = open(args.word_dict, 'r')
        word_dic = {}
        for line in fin:
            tokens = line.split()
            word_dic[''.join(tokens[1:])] = tokens[0]

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    mdic = torch.load(args.model_dic)
    model = Transformer(**mdic['params']).to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    if args.fp16: model.half()

    lm = None
    if args.lm_dic is not None:
        mdic = torch.load(args.lm_dic)
        lm = SeqLM(**mdic['params']).to(device)
        lm.load_state_dict(mdic['state'])

    reader = ScpStreamReader(args.data_scp, mean_sub=args.mean_sub,
                             fp16=args.fp16, downsample=args.downsample)
    reader.initialize()

    since = time.time()
    batch_size = args.batch_size
    fout = open(args.output, 'w')
    while True:
        src_seq, src_mask, utts = reader.read_batch_utt(batch_size)
        if len(utts) == 0: break
        with torch.no_grad():
            src_seq, src_mask = src_seq.to(device), src_mask.to(device)
            hypos, scores = beam_search(model, src_seq, src_mask, device, args.beam_size,
                                args.max_len, len_norm=args.len_norm, coverage=args.coverage,
                                lm=lm, lm_scale=args.lm_scale)
            hypos, scores = hypos.tolist(), scores.tolist()
            if args.format == 'ctm':
                write_ctm(hypos, scores, fout, utts, dic, word_dic, args.space)
            elif args.format == 'stm':
                write_stm(hypos, fout, utts, dic, word_dic, args.space)                
            else:
                write_text(hypos, fout, utts, dic, word_dic, args.space)
    fout.close()
    time_elapsed = time.time() - since
    print("  Elapsed Time: %.0fm %.0fs" % (time_elapsed // 60, time_elapsed % 60))

#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from pynn.io.kaldi_seq import ScpStreamReader, ScpBatchReader
from pynn.net.rel_tf import RelativeTransformer
from pynn.trainer.adam_seq2seq import train_model

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-scp', help='path to train scp', required=True)
parser.add_argument('--train-target', help='path to train target', required=True)
parser.add_argument('--valid-scp', help='path to validation scp', required=True)
parser.add_argument('--valid-target', help='path to validation target', required=True)

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--n-head', type=int, default=8)
parser.add_argument('--n-enc-head', type=int, default=0)
parser.add_argument('--n-enc', type=int, default=4)
parser.add_argument('--n-dec', type=int, default=4)
parser.add_argument('--d-input', type=int, default=80)
parser.add_argument('--d-model', type=int, default=512)
parser.add_argument('--d-inner-hid', type=int, default=1024)

parser.add_argument('--time-ds', help='downsample in time axis', type=int, default=1)
parser.add_argument('--use-cnn', help='use CNN filters', action='store_true')
parser.add_argument('--freq-kn', help='frequency kernel', type=int, default=3)
parser.add_argument('--freq-std', help='frequency stride', type=int, default=2)
parser.add_argument('--shared-kv', help='sharing key and value weights', action='store_true')
parser.add_argument('--shared-emb', help='sharing decoder embedding', action='store_true')
parser.add_argument('--attn-mode', help='encoder attention mode',  type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--emb-drop', type=float, default=0.0)
parser.add_argument('--enc-drop', type=float, default=0.0)
parser.add_argument('--dec-drop', type=float, default=0.0)
parser.add_argument('--label-smooth', type=float, default=0.1)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--teacher-force', type=float, default=1.0)

parser.add_argument('--downsample', help='concated frames', type=int, default=4)
parser.add_argument('--max-len', help='max sequence length', type=int, default=5000)
parser.add_argument('--max-utt', help='max utt per partition', type=int, default=4096)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--spec-drop', help='argument inputs', action='store_true')
parser.add_argument('--spec-bar', help='number of bars of spec-drop', type=int, default=2)
parser.add_argument('--time-stretch', help='argument inputs', action='store_true')
parser.add_argument('--time-win', help='time stretch window', type=int, default=10000)
parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--n-epoch', type=int, default=50)
parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=6000)
parser.add_argument('--n-const', help='constant steps', type=int, default=0)
parser.add_argument('--n-print', help='inputs per update', type=int, default=5000)
parser.add_argument('--batch', help='batch mode', action='store_true')
parser.add_argument('--b-input', help='inputs per batch', type=int, default=3000)
parser.add_argument('--b-update', help='characters per update', type=int, default=12000)
parser.add_argument('--shuffle', help='shuffle samples every epoch', action='store_true')
parser.add_argument('--lr', help='learning rate', type=float, default=2.0)

parser.add_argument('--loss-norm', help='per-token loss normalization', action='store_true')
parser.add_argument('--grad-norm', help='per-token grad normalization', action='store_true')
parser.add_argument('--fp16', help='fp16 or not', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    n_enc_head = args.n_head if args.n_enc_head==0 else args.n_enc_head
    model = RelativeTransformer(
        args.n_classes,
        d_input=args.d_input,
        d_model=args.d_model,
        d_inner=args.d_inner_hid,
        n_enc=args.n_enc,
        n_enc_head=n_enc_head,
        n_dec=args.n_dec,
        n_dec_head=args.n_head,
        time_ds=args.time_ds,
        use_cnn=args.use_cnn,
        freq_kn=args.freq_kn,
        freq_std=args.freq_std,
        shared_kv=args.shared_kv,
        shared_emb=args.shared_emb,
        attn_mode=args.attn_mode,
        dropout=args.dropout,
        emb_drop=args.emb_drop,
        enc_drop=args.enc_drop,
        dec_drop=args.dec_drop).to(device)

    ScpReader = ScpBatchReader if args.batch else ScpStreamReader
    tr_reader = ScpReader(args.train_scp, args.train_target, downsample=args.downsample, 
                                  max_len=args.max_len, max_utt=args.max_utt, shuffle=args.shuffle,
                                  mean_sub=args.mean_sub, fp16=args.fp16,
                                  spec_drop=args.spec_drop, spec_bar=args.spec_bar, 
                                  time_stretch=args.time_stretch, time_win=args.time_win)
    cv_reader = ScpStreamReader(args.valid_scp, args.valid_target, downsample=args.downsample, 
                                  mean_sub=args.mean_sub, max_len=args.max_len, max_utt=args.max_utt, fp16=args.fp16)

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'label_smooth': args.label_smooth,
           'weight_decay': args.weight_decay, 'teacher_force': args.teacher_force,
           'n_warmup': args.n_warmup, 'n_const': args.n_const,
           'b_input': args.b_input, 'b_update': args.b_update, 'n_print': args.n_print}
    datasets = (tr_reader, cv_reader)
    train_model(model, datasets, args.n_epoch, device, cfg,
                loss_norm=args.loss_norm, grad_norm=args.grad_norm, fp16=args.fp16)



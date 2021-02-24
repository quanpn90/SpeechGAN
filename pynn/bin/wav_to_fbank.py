#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import argparse
import multiprocessing
import numpy as np
from scipy.io import wavfile

from pynn.util import audio
from pynn.io import kaldi_io

def write_ark_thread(segs, out_ark, out_scp, args):
    fbank_mat = audio.filter_bank(args.sample_rate, args.nfft, args.fbank)
    cache_wav = ''
    
    for seg in segs:
        tokens = seg.split()
        
        if args.seg_info:
            wav, start, end = tokens[:3]
            seg_name = '%s-%06.f-%06.f' % (wav, float(start)*100, float(end)*100)
        else:
            if len(tokens) == 1: tokens.append(tokens[0])
            if len(tokens) == 2: tokens.extend(['0.0', '0.0'])
            seg_name, wav, start, end = tokens[:4]
        start, end = float(start), float(end)
        
        if args.wav_path is not None:
            wav = wav if wav.endswith('.wav') else wav + '.wav'
            wav = args.wav_path + '/' + wav
        if cache_wav != wav:
            if not os.path.isfile(wav):
                print('File %s does not exist' % wav)
                continue
            sample_rate, signal = wavfile.read(wav)
            if sample_rate != args.sample_rate:
                print('Wav %s is not in desired sample rate' % wav)
                continue
            cache_wav = wav

        start = int(start * sample_rate)
        end = -1 if end <= 0. else int(end * sample_rate)
        if start >= len(signal) or start >= end:
             print('Wrong segment %s' % seg_name)
             continue
        feats = audio.extract_fbank(signal[start:end], fbank_mat, sample_rate=sample_rate, nfft=args.nfft)
        if len(feats) > args.max_len or len(feats) < args.min_len:
            continue
        if args.mean_norm:
            feats = feats - feats.mean(axis=0, keepdims=True)
        if args.fp16:
            feats = feats.astype(np.float16) 

        dic = {seg_name: feats}
        kaldi_io.write_ark(out_ark, dic, out_scp, append=True)
    

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--seg-desc', help='segment description file', required=True)
parser.add_argument('--seg-info', help='append timestamps to the segment names', action='store_true')
parser.add_argument('--wav-path', help='path to wav files', type=str, default=None)
parser.add_argument('--sample-rate', help='sample rate', type=int, default=16000)
parser.add_argument('--fbank', help='number of filter banks', type=int, default=40)
parser.add_argument('--nfft', help='number of FFT points', type=int, default=256)
parser.add_argument('--max-len', help='maximum frames for a segment', type=int, default=10000)
parser.add_argument('--min-len', help='minimum frames for a segment', type=int, default=4)
parser.add_argument('--mean-norm', help='mean substraction', action='store_true')
parser.add_argument('--fp16', help='use float16 instead of float32', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='data')
parser.add_argument('--jobs', help='number of parallel jobs', type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    
    segs = [line.rstrip('\n') for line in open(args.seg_desc, 'r')]
    size = len(segs) // args.jobs
    jobs = []
    j = 0
    for i in range(args.jobs):
        l = len(segs) if i == (args.jobs-1) else j+size
        sub_segs = segs[j:l]
        j += size
        out_ark = '%s.%d.ark' % (args.output, i)
        out_scp = '%s.%d.scp' % (args.output, i)
         
        process = multiprocessing.Process(
                target=write_ark_thread, args=(sub_segs, out_ark, out_scp, args))
        process.start()
        jobs.append(process)
    
    for job in jobs: job.join()
    

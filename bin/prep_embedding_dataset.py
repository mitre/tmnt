# coding: utf-8

import argparse
import logging
import os
import random
import sys
import time
import io
import datetime
import multiprocessing
import json

import mxnet as mx
import numpy as np
import gluonnlp as nlp


from sentence_splitter import SentenceSplitter

parser = argparse.ArgumentParser('Reformat text files with each sentence/sample on a single line for faster processing')

parser.add_argument('--input_dir', type=str, help='Input .txt files')
parser.add_argument('--mode', type=str, help='json or txt/text', default='json')
parser.add_argument('--output_dir', type=str, help='Output directory with reformatted files')
parser.add_argument('--lowercase', action='store_true', help='Lowercase the data')

args = parser.parse_args()


if __name__ == '__main__':
    files = os.listdir(args.input_dir)
    splitter = SentenceSplitter('en')
    cnt = 0
    total_lines = 0
    ostr = None
    json_mode = True if args.mode == 'json' else False
    for f in files:
        in_file = os.path.join(args.input_dir, f)
        if os.path.isfile(in_file):
            if total_lines > 1000 or cnt < 1:
                cnt += 1
                ofile = os.path.join(args.output_dir, ('train_emb'+ str(cnt)+ '.txt'))
                if ostr:
                    ostr.close()
                ostr = open(ofile, 'w')
                total_lines = 0
            if json_mode:
                with open(in_file, 'r') as fp:
                    for line in fp:
                        js = json.loads(line)
                        data = js['text']
                        sents = splitter.split(data)
                        for s in sents:
                            if len(s) > 10:
                                s = s.lower() if args.lowercase else s
                                ostr.write(s)
                                ostr.write('\n')
                                total_lines += 1
            else:
                with open(in_file, 'r') as fp:
                    data = fp.read().replace('\n', ' ')
                    sents = splitter.split(data)
                    for s in sents:
                        if len(s) > 10:
                            s = s.lower() if args.lowercase else s
                            ostr.write(s)
                            ostr.write('\n')
                            total_lines += 1

                
        

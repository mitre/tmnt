# coding: utf-8

import os, sys
import argparse

from tmnt.models.seq_bow.train import train_main

parser = argparse.ArgumentParser(description='Train a Transformer-based Variational Encoder-Decoder on Context-aware Encodings with BOW decoding')

parser.add_argument('--input_file', type=str, help='Directory containing a JSON list file representing the input data')
parser.add_argument('--use_gpu',action='store_true', help='Use GPU(s) if available', default=False)
parser.add_argument('--save_dir',type=str, help='Target directory for trained model parameters', default='seqvae_exp_logs')
parser.add_argument('--wd_embed_dim',type=int, help='Word embedding dimension (for output)', default=256)
parser.add_argument('--kld_wt',type=float, help='Weight of the KL divergence term in variational loss', default=1.0)
parser.add_argument('--sent_size',type=int, help='Fixed/max length of sentence (zero padded); should be power of 2', default=32)
parser.add_argument('--model_dir', type=str, help='Directory for final saved model files', default=None)
parser.add_argument('--weight_decay', type=float, help='Learning weight decay', default=0.00001)
parser.add_argument('--warmup_ratio', type=float, help='Percentage of training steps after which decay begins (default 0.1)', default=0.1)
parser.add_argument('--log_interval', type=int, help='Number of batches after which loss and reconstruction examples will be logged', default=20)
parser.add_argument('--save_interval', type=int, help='Number of EPOCHs after which model checkpoints will be saved', default=20)
parser.add_argument('--offset_factor', type=float, help='Adjusts offset for LR decay; values < 1 are faster', default=1.0)
parser.add_argument('--min_lr', type=float, help='Absolute minimum LR', default=1e-7)
parser.add_argument('--wd_temp', type=float, help='Temperature coefficient for output embedding' ,default=0.01)
parser.add_argument('--use_bert', action='store_true', help='Use BERT base as the encoder (and fine-tune)')
parser.add_argument('--embedding_source', type=str, help='Word embedding source to use (if not using BERT)', default=None)
parser.add_argument('--json_text_key', type=str, help='Assume json list format and select text using this key', default=None)
parser.add_argument('--bow_vocab_file', type=str, help='Vocabulary for BOW in decoder', default=None)
parser.add_argument('--config', type=str, help='JSON-formatted configuration file', default=None)

args = parser.parse_args()

train_main(args)

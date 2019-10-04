# coding: utf-8

import os, sys
import argparse

from tmnt.seq_vae.train import train_main
from tmnt.seq_vae.ar_seq_models import test_ar

parser = argparse.ArgumentParser(description='Train a Transformer-based Variational AutoEncoder on Context-aware Encodings')

parser.add_argument('--input_file', type=str, help='Directory containing a RecordIO file representing the input data')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, bertadam)', default='bertadam')
parser.add_argument('--gen_lr', type=float, help='General learning rate', default=0.00001)
parser.add_argument('--gpus',type=str, help='GPU device ids', default='')
parser.add_argument('--save_dir',type=str, help='Target directory for trained model parameters', default='seqvae_exp_logs')
parser.add_argument('--batch_size',type=int, help='Training batch size', default=8)
parser.add_argument('--latent_dim',type=int, help='Encoder dimensionality', default=256)
parser.add_argument('--num_units', type=int, help='Hidden dimensions in Transformer Decoder', default=256)
parser.add_argument('--num_heads', type=int, help='Number of heads in Transformer self-attention', default=8)
parser.add_argument('--hidden_size', type=int, help='Size of hidden dim in Transformer blocks', default=512)
parser.add_argument('--transformer_layers',type=int, help='Decoder transformer layers', default=6)
parser.add_argument('--wd_embed_dim',type=int, help='Word embedding dimension (for output)', default=256)
parser.add_argument('--kld_wt',type=float, help='Weight of the KL divergence term in variational loss', default=1.0)
parser.add_argument('--sent_size',type=int, help='Fixed/max length of sentence (zero padded); should be power of 2', default=16)
parser.add_argument('--model_dir', type=str, help='Directory for final saved model files', default=None)
parser.add_argument('--weight_decay', type=float, help='Learning weight decay', default=0.00001)
parser.add_argument('--warmup_ratio', type=float, help='Percentage of training steps after which decay begins (default 0.1)', default=0.1)
parser.add_argument('--log_interval', type=int, help='Number of batches after which loss and reconstruction examples will be logged', default=20)
parser.add_argument('--save_interval', type=int, help='Number of EPOCHs after which model checkpoints will be saved', default=20)
parser.add_argument('--offset_factor', type=float, help='Adjusts offset for LR decay; values < 1 are faster', default=1.0)
parser.add_argument('--min_lr', type=float, help='Absolute minimum LR', default=1e-7)
parser.add_argument('--wd_temp', type=float, help='Temperature coefficient for output embedding' ,default=0.01)
parser.add_argument('--latent_dist', type=str, help='Latent distribution', default='vmf')
parser.add_argument('--kappa', type=float, help='vMF distribution kappa (concentration) parameter', default=100.0)
parser.add_argument('--use_bert', action='store_true', help='Use BERT base as the encoder (and fine-tune)')
parser.add_argument('--embedding_source', type=str, help='Word embedding source to use (if not using BERT)', default='glove.6B.50d')
parser.add_argument('--max_vocab_size', type=int, help='Maximum size of vocabulary (if not using BERT)', default=20000)
parser.add_argument('--json_text_key', type=str, help='Assume json list format and select text using this key', default=None)
parser.add_argument('--label_smoothing_epsilon', type=float, help='Label smoothing epsilon value', default=0.1)
parser.add_argument('--ar_decoder', action='store_true', help='Use auto-regressive transformer decoder', default=False)


args = parser.parse_args()

if args.ar_decoder:
    test_ar(args)
else:
    train_main(args)


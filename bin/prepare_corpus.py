# coding: utf-8

import os, sys
import argparse

from tmnt.preprocess import get_sparse_vecs, get_sparse_vecs_txt
from tmnt.preprocess.json_to_csr import set_text_key, set_min_doc_size

parser = argparse.ArgumentParser('Prepare a training and validation/test dataset for topic model training')

parser.add_argument('--tr_input_dir', type=str, help='Directory of training files (json batches)')
parser.add_argument('--tst_input_dir', type=str, help='Directory of test files (json batches)', default=None)
parser.add_argument('--file_pat', type=str, help='File pattern')
parser.add_argument('--tr_vec_file', type=str, help='Output file with training documents in sparse vector format')
parser.add_argument('--tst_vec_file', type=str, help='Output file with test documents in sparse vector format', default=None)
parser.add_argument('--vocab_size', type=int, help='Size of the vocabulary to construct', default=2000)
parser.add_argument('--vocab_file', type=str, help='File for resulting vocabulary')
parser.add_argument('--full_vocab_histogram', type=str, help='Optional output of entire histogram', default=None)
parser.add_argument('--txt_mode', action='store_true', help='Assume txt file input (1 document per file)')
parser.add_argument('--json_text_key', type=str, help='Key for json field containing document text (default is \'text\')', default='text')
parser.add_argument('--min_doc_length', type=int, help='Minimum document length (in tokens)', default=10)

args = parser.parse_args()

if args.txt_mode:
    vocab = get_sparse_vecs_txt(args.tr_vec_file, args.vocab_file, args.tr_input_dir, args.vocab_size, full_histogram_file=args.full_vocab_histogram,
                                pat=args.file_pat)
    _ = get_sparse_vecs_txt(args.tst_vec_file, args.vocab_file, args.tst_input_dir, i_vocab=vocab, pat=args.file_pat)
else:
    set_text_key(args.json_text_key)
    set_min_doc_size(args.min_doc_length)
    vocab = get_sparse_vecs(args.tr_vec_file, args.vocab_file, args.tr_input_dir, args.vocab_size, full_histogram_file=args.full_vocab_histogram,
                            pat=args.file_pat)
    if args.tst_input_dir and args.tst_vec_file:
        _ = get_sparse_vecs(args.tst_vec_file, args.vocab_file, args.tst_input_dir, i_vocab=vocab, pat=args.file_pat)

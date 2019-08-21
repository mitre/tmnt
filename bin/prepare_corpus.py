# coding: utf-8

import os, sys
import argparse

from tmnt.preprocess.vectorizer import JsonVectorizer, TextVectorizer

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
parser.add_argument('--custom_stop_words', type=str, help='Custom stop-word file (one word per line)', default=None)

args = parser.parse_args()

vectorizer = TextVectorizer(min_doc_size=args.min_doc_length) if args.txt_mode else JsonVectorizer(text_key=args.json_text_key, min_doc_size=args.min_doc_length)
vocab = vectorizer.get_sparse_vecs(args.tr_vec_file, args.vocab_file, args.tr_input_dir, args.vocab_size, full_histogram_file=args.full_vocab_histogram,
                            pat=args.file_pat)
if args.tst_input_dir and args.tst_vec_file:
    _ = vectorizer.get_sparse_vecs(args.tst_vec_file, args.vocab_file, args.tst_input_dir, i_vocab=vocab, pat=args.file_pat)

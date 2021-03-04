# coding: utf-8

import os, sys
import argparse
import logging
from tmnt.utils.log_utils import logging_config

from tmnt.preprocess.vectorizer import TMNTVectorizer

parser = argparse.ArgumentParser('Prepare a training and validation/test dataset for topic model training with a VED model')

parser.add_argument('--tr_input_file', type=str, help='FILE of training documents')
parser.add_argument('--val_input_file', type=str, help='FILE of validation test documents', default=None)
parser.add_argument('--tst_input_file', type=str, help='FILE of held out test files', default=None)
parser.add_argument('--json_out_dir', type=str, help='Create a new JSON file directory for training with vectors added as a field in this target directory')
parser.add_argument('--file_pat', type=str, help='File pattern', default='*.json')
parser.add_argument('--vocab_size', type=int, help='Size of the vocabulary to construct', default=2000)
parser.add_argument('--vocab_file', type=str, help='File for resulting vocabulary')
parser.add_argument('--full_vocab_histogram', type=str, help='Optional output of entire histogram', default=None)
parser.add_argument('--json_text_key', type=str, help='Key for json field containing document text (default is \'text\')', default='text')
parser.add_argument('--json_label_key', type=str, help='Key for json field containing label (default is None)', default=None)
parser.add_argument('--min_doc_length', type=int, help='Minimum document length (in tokens)', default=10)
parser.add_argument('--custom_stop_words', type=str, help='Custom stop-word file (one word per line)', default=None)
parser.add_argument('--label_prefix_chars', type=int, help='Use first N characters of label', default=-1)
parser.add_argument('--str_encoding', type=str, help='String/file encoding to use', default='utf-8')
parser.add_argument('--log_dir', type=str, help='Logging directory', default='.')

args = parser.parse_args()

if __name__ == '__main__':
    logging_config(folder=args.log_dir, name='vectorizer', level='info')
    if args.vocab_file is None:
        raise Exception("Vocabulary output file name/path must be provided")
    vectorizer = \
           TMNTVectorizer(text_key=args.json_text_key, custom_stop_word_file=args.custom_stop_words, label_key=args.json_label_key,
                            min_doc_size=args.min_doc_length, label_prefix=args.label_prefix_chars,
                            json_out_dir=args.json_out_dir, vocab_size = args.vocab_size,
                            encoding=args.str_encoding)
    vectorizer.fit_transform_in_place_json(args.tr_input_file)
    vectorizer.write_vocab(args.vocab_file)
    if args.val_input_file:
        vectorizer.transform_in_place_json(args.val_input_file)
    if args.tst_input_file:
        vectorizer.transform_in_place_json(args.tst_input_file)
    

# coding: utf-8

import os, sys, io
import argparse
import logging
import json
from tmnt.utils.log_utils import logging_config
from tmnt.preprocess.vectorizer import TMNTVectorizer

parser = argparse.ArgumentParser('Prepare a training and validation/test dataset for topic model training')

parser.add_argument('--tr_input', type=str, help='Directory of training files (jsonl batches) or SINGLE file')
parser.add_argument('--val_input', type=str, help='Directory of validation test files (jsonl batches) or SINGLE file', default=None)
parser.add_argument('--tst_input', type=str, help='Directory of held out test files (jsonl batches) or SINGLE file', default=None)
parser.add_argument('--file_pat', type=str, help='File pattern', default='*.json')
parser.add_argument('--tr_vec_file', type=str, help='Output file with training documents in sparse vector format')
parser.add_argument('--val_vec_file', type=str, help='Output file with test validation documents in sparse vector format', default=None)
parser.add_argument('--tst_vec_file', type=str, help='Output file with heldout test documents in sparse vector format', default=None)
parser.add_argument('--vocab_size', type=int, help='Size of the vocabulary to construct', default=2000)
parser.add_argument('--vocab_file', type=str, help='File for resulting vocabulary')
parser.add_argument('--full_vocab_histogram', type=str, help='Optional output of entire histogram', default=None)
parser.add_argument('--json_text_key', type=str, help='Key for json field containing document text (default is \'text\')', default='text')
parser.add_argument('--json_label_key', type=str,
                    help='Key for json field containing label (default is None). Only set if labels always available',
                    default=None)
parser.add_argument('--label_map', type=str, help='JSON object to file with mapping between labels and indices', default=None)
parser.add_argument('--json_out_dir', type=str, help='Create a new JSON list file with vectors added as a field in this target directory')
parser.add_argument('--min_doc_length', type=int, help='Minimum document length (in tokens)', default=10)
parser.add_argument('--custom_stop_words', type=str, help='Custom stop-word file (one word per line)', default=None)
parser.add_argument('--label_prefix_chars', type=int, help='Use first N characters of label', default=-1)
parser.add_argument('--str_encoding', type=str, help='String/file encoding to use', default='utf-8')
parser.add_argument('--log_dir', type=str, help='Logging directory', default='.')

args = parser.parse_args()

if __name__ == '__main__':
    logging_config(folder=args.log_dir, name='vectorizer', level=logging.INFO)
    if args.vocab_file is None:
        raise Exception("Vocabulary output file name/path must be provided")
    if (args.tr_vec_file is None) or (args.tr_input is None):
        raise Exception("Training directory and output vector file must be provided")
    vectorizer = \
        TMNTVectorizer(text_key=args.json_text_key, custom_stop_word_file=args.custom_stop_words,
                       label_key=args.json_label_key,
                       min_doc_size=args.min_doc_length, label_prefix=args.label_prefix_chars,
                       file_pat=args.file_pat,
                       vocab_size=args.vocab_size,
                       json_out_dir=args.json_out_dir,
                       encoding=args.str_encoding)
    tr_X, tr_y = \
        vectorizer.fit_transform_json_dir(args.tr_input) if os.path.isdir(args.tr_input) else vectorizer.fit_transform_json(args.tr_input)
    vectorizer.write_to_vec_file(tr_X, tr_y, args.tr_vec_file)
    vectorizer.write_vocab(args.vocab_file)
    if args.val_input and args.val_vec_file:
        val_X, val_y = \
            vectorizer.transform_json_dir(args.val_input) if os.path.isdir(args.val_input) else vectorizer.fit_transform_json(args.val_input)
        vectorizer.write_to_vec_file(val_X, val_y, args.val_vec_file)
    if args.tst_input and args.tst_vec_file:
        tst_X, tst_y = \
            vectorizer.transform_json_dir(args.tst_input) if os.path.isdir(args.tst_input) else vectorizer.fit_transform_json(args.tst_input)
        vectorizer.write_to_vec_file(tst_X, tst_y, args.tst_vec_file)
    if args.label_map:
        with io.open(args.label_map, 'w') as fp:
            fp.write(json.dumps(vectorizer.label_map, indent=4))

                                       

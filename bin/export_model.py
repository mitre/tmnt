# coding: utf-8

import os, io
import argparse
import funcy

from tmnt.bow_vae.runtime import BowNTMInference

parser = argparse.ArgumentParser()

parser.add_argument('--param_file', type=str, help='Parameter file')
parser.add_argument('--config_file', type=str, help='Config file')
parser.add_argument('--vocab_file', type=str, help='Vocab (.json) file')
parser.add_argument('--vec_file', type=str, help='Sparse vector file')
parser.add_argument('--json_output_file', type=str, help='JSON output file')
parser.add_argument('--html_vis', type=str, help='PyLDAVis HTML file', default=None)

args = parser.parse_args()

infer = BowNTMInference(args.param_file, args.config_file, args.vocab_file)

full_model_dict = infer.export_full_model_inference_details(args.vec_file, args.json_output_file)

if args.html_vis:
    import pyLDAvis
    opts = funcy.merge(full_model_dict, {'mds': 'mmds'})
    vis_data = pyLDAvis.prepare(**opts)
    pyLDAvis.save_html(vis_data, args.html_vis)

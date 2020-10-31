# coding: utf-8

import os, io
import argparse
import funcy
import json
from pathlib import Path
import warnings

from tmnt.inference import BowVAEInferencer

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=Path, help='Directory with trained model files')
parser.add_argument('--vec_file', type=str, help='Sparse vector file')
parser.add_argument('--json_output_file', type=str, help='JSON output file')
parser.add_argument('--json_simple', type=str, help='JSON output file')
parser.add_argument('--html_vis', type=str, help='PyLDAVis HTML file', default=None)
parser.add_argument('--str_encoding', type=str, help='String/file encoding', default='utf-8')

args = parser.parse_args()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    os.environ["MXNET_STORAGE_FALLBACK_LOG_VERBOSE"] = "0"
    param_file, config_file, vocab_file = \
        args.model_dir / "model.params", args.model_dir / "model.config", args.model_dir / "vocab.json"
    infer = BowVAEInferencer(param_file, config_file, vocab_file)

    if args.json_output_file:
        full_model_dict = infer.export_full_model_inference_details(args.vec_file, args.json_output_file)
    if args.html_vis:
        import pyLDAvis
        full_model_dict = infer.get_pyldavis_details(args.vec_file)
        opts = funcy.merge(full_model_dict, {'mds': 'mmds'})
        vis_data = pyLDAvis.prepare(**opts)
        pyLDAvis.save_html(vis_data, args.html_vis)

    if args.json_simple:
        w_pr, _, _, _ = infer.get_model_details(args.vec_file)
        k, n = w_pr.shape
        vocab = infer.vocab
        d = {}
        for i in range(k):
            tn = "topic_"+str(i)
            vl = []
            for j in range(n):
                vl.append((vocab._idx_to_token[j], float(w_pr[i,j].asscalar())))
            d[tn] = vl
        with io.open(args.json_simple, 'w', encoding=args.str_encoding) as fp:
            json.dump(d, fp, indent=4)

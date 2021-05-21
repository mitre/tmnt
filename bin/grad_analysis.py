#!/usr/bin/env python3

import math
import argparse
import io
import os
import mxnet as mx

from tmnt.inference import BowVAEInferencer
from tmnt.data_loading import file_to_data, load_vocab
from tmnt.eval_npmi import NPMI, EvaluateNPMI
from tmnt.utils.ngram_helpers import BigramReader
from tmnt.data_loading import DataIterLoader, SparseMatrixDataIter
import gluonnlp as nlp

from itertools import combinations

import umap
from pathlib import Path


import numpy as np

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Variational AutoEncoder topic model')
    parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
    parser.add_argument('--test_file', type=str, required=True, help='file in sparse vector format')    
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file associated with sparse vector data')
    parser.add_argument('--model_dir', type=Path,
                        help='The directory where the params, specs, and vocab should be found.')
    parser.add_argument('--plot_file', type=str, help='Output plot')
    parser.add_argument('--words_per_topic', type=int, help='Number of terms per topic to output', default=10)
    parser.add_argument('--override_top_k_terms', type=str, help='File of topic terms to use instead of those from model', default=None)
    parser.add_argument('--encoder_terms', type=str, help='File output of encoder terms', default=None)
    return parser

def read_vector_file(file):
    labels = []
    docs = []
    with open(file) as f:
        for line in map(str.strip, f):
            label, *words = line.split()
            labels.append(int(label))
            docs.append(list(map(lambda t: int(t.split(":")[0]), words)))
    return labels, docs

def evaluate(inference, data_loader, total_words, ctx=mx.cpu()):
    total_rec_loss = 0
    for i, (data,_) in enumerate(data_loader):
        data = data.as_in_context(ctx)
        _, rec_loss, _, log_out = inference.model(data)
        total_rec_loss += rec_loss.sum().asscalar()
    perplexity = math.exp(total_rec_loss / total_words)
    return perplexity


def get_top_k_terms_from_file(in_file):
    top_k_terms = []
    with io.open(in_file, 'r') as fp:
        for l in fp:
            ts = [t.strip() for t in l.split(',')]
            top_k_terms.append(ts)
    return top_k_terms


os.environ["MXNET_STORAGE_FALLBACK_LOG_VERBOSE"] = "0"

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    verbose = False ### XXX - add as argument
    inference_model = BowVAEInferencer.from_saved(model_dir=args.model_dir,
                                                  ctx=mx.cpu() if args.gpu < 0 else mx.gpu(args.gpu))
    
    if args.override_top_k_terms:
        top_k_words_per_topic = get_top_k_terms_from_file(args.override_top_k_terms)
        tst_csr, _, _, _ = file_to_data(args.test_file, len(inference_model.vocab))
        top_k_words_per_topic_ids = [ [ inference_model.vocab[t] for t in t_set ]  for t_set in top_k_words_per_topic ]
        npmi_eval = EvaluateNPMI(top_k_words_per_topic_ids)
        test_npmi = npmi_eval.evaluate_csr_mat(tst_csr)
        print("**** Test NPMI = {} *******".format(test_npmi))
        exit(0)


    if args.plot_file: # get UMAP embedding visualization
        import matplotlib.pyplot as plt
        encoded, labels = inference_model.encode_vec_file(args.test_file)
        encodings = np.array([doc.asnumpy() for doc in encoded])
        print("There are {0} labels and {1} encodings".format(len(labels), len(encodings)))
        umap_model = umap.UMAP(n_neighbors=4, min_dist=0.5, metric='euclidean')
        embeddings = umap_model.fit_transform(encodings)
        plt.scatter(*embeddings.T, c=labels, s=0.2, alpha=0.7, cmap='coolwarm')
        plt.savefig(args.plot_file, dpi=1000)

    top_k_words_per_topic = inference_model.get_top_k_words_per_topic(args.words_per_topic)        
    for i in range(len(top_k_words_per_topic)):
        print("Topic {}: {}".format(i, top_k_words_per_topic[i]))

    top_k_words_per_topic_ids = [ [ inference_model.vocab[t] for t in t_set ]  for t_set in top_k_words_per_topic ]

    npmi_eval = EvaluateNPMI(top_k_words_per_topic_ids)
    tst_csr, _, _, _ = file_to_data(args.test_file, len(inference_model.vocab))
    test_npmi = npmi_eval.evaluate_csr_mat(tst_csr)
    print("**** Test NPMI = {} *******".format(test_npmi))
    if args.encoder_terms:
        sample_size = min(2000, tst_csr.shape[0])
        batch_size  = min(8, tst_csr.shape[0])
        dataloader = DataIterLoader(SparseMatrixDataIter(tst_csr[:sample_size], None, batch_size=batch_size, last_batch_handle='pad', shuffle=False))
        top_k_words_encoder = inference_model.get_top_k_words_per_topic_encoder(args.words_per_topic, dataloader, sample_size=sample_size)
        for i in range(len(top_k_words_encoder)):
            print("Encoder topic {}: {}".format(i, top_k_words_encoder[i]))
        top_k_words_encoder_per_topic_ids = [ [ inference_model.vocab[t] for t in t_set ]  for t_set in top_k_words_encoder ]
        enc_npmi_eval = EvaluateNPMI(top_k_words_encoder_per_topic_ids)
        print("**** Test ENCODER NPMI = {} *******".format(enc_npmi_eval.evaluate_csr_mat(tst_csr)))
        details = inference_model.model.get_ordered_terms_per_item(dataloader, sample_size=sample_size)
        #encodings = inference_model.encode_data(tst_csr[:sample_size], None, use_probs=True)
        encodings = inference_model.encode_data(tst_csr[:sample_size], None, use_probs=False)
        #encodings = inference_model.encode_vec_file(args.test_file)[0]
        ##
        details_np = np.array(details).swapaxes(0,1)
        print("Encodings for docs")
        for i,e in enumerate(encodings):
            print("Encodings doc {}: {}".format(i, e))
        for topic_id in range(inference_model.model.n_latent):
            print("***** ANALYSIS ON TOPIC {} *****".format(topic_id))
            details_i = np.array(details[topic_id][:sample_size])
            #selected_doc_ids_tp = np.where(np.isin((-np.array(encodings)).argsort(axis=1)[:,:3], topic_id).max(axis=1) > 0)[0]
            #print("Selected doc ids topic prominence: {}".format(selected_doc_ids_tp))
            for t in top_k_words_encoder[topic_id]:
                t_id = inference_model.vocab[t]
                print("t = {} ==> id = {}".format(t, t_id))
                selected_docs_having_term = np.where(tst_csr[:sample_size].toarray()[:,t_id] > 0)[0]
                print("Docs having term {} ==> {}".format(t, selected_docs_having_term))
                #selected_doc_ids = np.intersect1d(selected_doc_ids_tp, selected_docs_having_term)
                selected_doc_ids = selected_docs_having_term
                t_id_details = details_i[selected_doc_ids, t_id]
                if np.size(t_id_details) > 0:
                    print("Encoder info on: {} ==> Mean: {} Std: {}, Min: {}, Max: {}"
                          .format(t, t_id_details.mean(), t_id_details.std(), t_id_details.min(),  t_id_details.max()))
                else:
                    print("Encoder info on: {} ==> NO DATA".format(t))
            print("For DECODER terms:")
            for t in top_k_words_per_topic[topic_id]:
                t_id = inference_model.vocab[t]
                selected_docs_having_term = np.where(tst_csr[:sample_size].toarray()[:,t_id] > 0)[0]
                #selected_doc_ids = np.intersect1d(selected_doc_ids_tp, selected_docs_having_term)
                selected_doc_ids = selected_docs_having_term
                t_id_details = details_i[selected_doc_ids, t_id]
                if np.size(t_id_details) > 0:
                    num_negative = (t_id_details[t_id_details < 0]).shape[0]
                    print("Encoder info on: {} ==> Mean: {} Std: {}, Min: {}, Max: {}, Num Negative: {} (Support: {})"
                          .format(t, t_id_details.mean(), t_id_details.std(), t_id_details.min(),  t_id_details.max(), num_negative, np.size(t_id_details)))
                else:
                    print("Encoder info on {} ==> NO DATA".format(t))
                t_grad0 = details_np[0,:,t_id]
                print("t_grad[doc0] {} ==> {}".format(t, t_grad0))
                t_grad1 = details_np[1,:,t_id]
                print("t_grad[doc1] {} ==> {}".format(t, t_grad1))
            #print("Top terms for each document")
            #for d in range(details_i.shape[0]):
            #    doc_term_scores = details_i[d]
            #    sorted_scores = (- doc_term_scores).argsort()
            #    terms = [inference_model.vocab.idx_to_token[i] for i in list(sorted_scores[:10]) ]
            #    print("Top terms: {}".format(' '.join(terms)))
            ##
            ## for each of the top decoder terms --  get lowest ranked position in the
            ranks = {}
            cnt_rank = 0
            for t in top_k_words_per_topic[topic_id]:
                t_id = inference_model.vocab[t]
                for d in range(details_i.shape[0]):
                    if topic_id in list((-encodings[d]).argsort())[:3]: ## if topic_id is in the top 3 topics
                        cnt_rank += 1
                        doc_term_scores = details_i[d]
                        sorted_scores = (- doc_term_scores).argsort()
                        cur_ranks = ranks.get(t) or []
                        ranks[t] = [np.where(sorted_scores == t_id)[0][0] + 1] + cur_ranks
            print("***********************")
            print("***Rank Analysis Topic {}*** (With {} selected docs based on topic prominence)".format(topic_id, cnt_rank/len(top_k_words_per_topic[topic_id])))
            for t in ranks:
                ra = np.array(ranks[t])
                print('For {} ==> best: {}, worst: {}, avg: {}'.format(t, ra.min(), ra.max(), ra.mean()))
            print("\n\n\n")
                
    exit(0)




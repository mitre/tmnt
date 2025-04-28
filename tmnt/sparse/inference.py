from tmnt.inference import SeqVEDInferencer
from scipy.sparse import csr_matrix
import numpy as np
from typing import List, Tuple
from tmnt.distribution import ConceptLogisticGaussianDistribution
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, IterableDataset
import tqdm
from datasets.arrow_writer import ArrowWriter
from tmnt.inference import SeqVEDInferencer
import io, json

__all__ = ['batch_process_to_arrow']

def csr_to_indices_data(csr_mat):
    return [ (csr_mat.getrow(ri).indices, csr_mat.getrow(ri).data) for ri in range(csr_mat.shape[0]) ]

def batch_process_to_arrow(model_path, json_input_texts, output_db_path, max_seq_len=512, device='cuda', batch_size=200, json_txt_key='text'):

    inferencer = SeqVEDInferencer.from_saved(model_path, max_length=max_seq_len, device=device)
    def encode_batch(txt_batch):
        tokenization_result = inferencer.prep_text(txt_batch)
        llm_out = inferencer.model.llm(tokenization_result['input_ids'].to(inferencer.device), 
                                            tokenization_result['attention_mask'].to(inferencer.device))
        cls_vecs = inferencer.model._get_embedding(llm_out, tokenization_result['attention_mask'].to(inferencer.device))
        raw_concepts = inferencer.model.latent_distribution.get_sparse_encoding(cls_vecs).cpu().detach()
        mu_emb = inferencer.model.latent_distribution.get_mu_encoding(cls_vecs)
        encs : List[List[float]] = cls_vecs.cpu().detach().tolist() 
        sparse_concepts : List[Tuple[List[int], List[float]]] = csr_to_indices_data(csr_matrix(raw_concepts))
        topic_embeddings : List[List[float]] = mu_emb.cpu().detach().tolist()
        print("Lengths: {}, {}, {}, {}".format(len(txt_batch), len(encs), len(sparse_concepts), len(topic_embeddings)))
        return zip(txt_batch, encs, sparse_concepts, topic_embeddings)
    
    def write_encodings(writer: ArrowWriter, txt_enc_pairs):
        for (text, embedding, sparse_indices_and_data, topic_embedding) in txt_enc_pairs:
            writer.write({'text': text, 'embedding': embedding, 'indices': sparse_indices_and_data[0], 
                          'values': sparse_indices_and_data[1], 'topic_embedding': topic_embedding})

    with io.open(json_input_texts) as fp:
        with ArrowWriter(path=output_db_path) as writer:
            txt_batch = []
            for l in fp:
                js = json.loads(l)
                txt_batch.append(js[json_txt_key])
                if len(txt_batch) >= batch_size:
                    encodings = encode_batch(txt_batch)
                    write_encodings(writer, encodings)
                    txt_batch = []
            if len(txt_batch) > 0:
                encodings = encode_batch(txt_batch)
                write_encodings(writer, encodings) 
            writer.finalize() 
    


import torch
from torch.utils.data import DataLoader
from datasets import Dataset, IterableDataset
import tqdm
from datasets.arrow_writer import ArrowWriter
from tmnt.inference import SeqVEDInferencer
import io, json
from tmnt.sparse.modeling import BaseAutoencoder
from typing import List

__all__ = ['ActivationsStore', 'build_activation_store', 'build_activation_store_batching', 'train_sparse_encoder_decoder']

class ActivationsStore:
    def __init__(
        self,
        cfg: dict,
    ):
        self.device = cfg["device"]
        self.activation_path = cfg["activation_path"]
        shuffle = cfg.get("shuffle_data", False)
        #self.dataset = Dataset.from_file(self.activation_path).with_format('torch', device=self.device)
        self.dataset = Dataset.from_file(self.activation_path).select_columns(['data']).shuffle(seed=42).with_format('torch', device=self.device)
        self.dataloader = DataLoader(self.dataset, 
                batch_size=cfg["batch_size"], shuffle=shuffle)
        self.dataloader_iter = iter(self.dataloader)
        self.cfg = cfg

    def next_batch(self):
        try:
            return next(self.dataloader_iter)['data']
        except (StopIteration, AttributeError):
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)['data']

def build_activation_store(json_input_texts, emb_model_path, arrow_output, max_seq_len=512, json_txt_key='text', device='cpu'):

    inferencer = SeqVEDInferencer.from_saved(emb_model_path, max_length=max_seq_len, device=device)
    with io.open(json_input_texts) as fp:
        with ArrowWriter(path=arrow_output) as writer:
            for l in fp:
                js = json.loads(l)
                tokenization_result = inferencer.prep_text(js[json_txt_key])
                llm_out = inferencer.model.llm(tokenization_result['input_ids'].to(inferencer.device), 
                                            tokenization_result['attention_mask'].to(inferencer.device))
                cls_vec = inferencer.model._get_embedding(llm_out, tokenization_result['attention_mask'].to(inferencer.device))
                enc : List[float] = cls_vec.cpu().detach()[0].tolist() 
                writer.write({'data': enc})
            writer.finalize()

def build_activation_store_batching(json_input_texts, emb_model_path, arrow_output, max_seq_len=512, batch_size=42, json_txt_key='text', device='cpu'):
    inferencer = SeqVEDInferencer.from_saved(emb_model_path, max_length=max_seq_len, device=device)
    def encode_batch(txt_batch):
        tokenization_result = inferencer.prep_text(txt_batch)
        llm_out = inferencer.model.llm(tokenization_result['input_ids'].to(inferencer.device), 
                                            tokenization_result['attention_mask'].to(inferencer.device))
        cls_vec = inferencer.model._get_embedding(llm_out, tokenization_result['attention_mask'].to(inferencer.device))
        encs : List[List[float]] = cls_vec.cpu().detach().tolist() 
        return zip(txt_batch, encs)
    
    def write_encodings(writer: ArrowWriter, txt_enc_pairs):
        for (t, e) in txt_enc_pairs:
            writer.write({'text': t, 'data': e})

    with io.open(json_input_texts) as fp:
        with ArrowWriter(path=arrow_output) as writer:
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


def train_sparse_encoder_decoder(sed: BaseAutoencoder, activation_store: ActivationsStore, cfg: dict):
    num_batches = cfg["num_samples"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sed.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    for i in pbar:
        batch = activation_store.next_batch()
        sed_output = sed(batch)

        loss = sed_output["loss"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dead": f"{sed_output['num_dead_features']:.4f}", "L0": f"{sed_output['l0_norm']:.4f}", "L2": f"{sed_output['l2_loss']:.4f}", "L1": f"{sed_output['l1_loss']:.4f}", "L1_norm": f"{sed_output['l1_norm']:.4f}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sed.parameters(), cfg["max_grad_norm"])
        sed.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()


    

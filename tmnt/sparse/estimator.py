import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import tqdm
from datasets.arrow_writer import ArrowWriter
from tmnt.inference import SeqVEDInferencer
import io, json
from tmnt.sparse.modeling import BaseAutoencoder
from typing import List

class ActivationsStore:
    def __init__(
        self,
        cfg: dict,
    ):
        self.device = cfg["device"]
        self.activation_path = cfg["activation_path"]
        self.dataloader = DataLoader(Dataset.from_file(self.activation_path).with_format('torch', device=self.device), 
                batch_size=cfg["batch_size"], shuffle=True)
        self.dataloader_iter = iter(self.dataloader)
        self.cfg = cfg

    def next_batch(self):
        try:
            return next(self.dataloader_iter)['data']
        except (StopIteration, AttributeError):
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)['data']

def build_activation_store(json_input_texts, emb_model_path, arrow_output, json_txt_key='text', device='cpu'):

    inferencer = SeqVEDInferencer.from_saved(emb_model_path, device=device)
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


def run_estimator_test():
    from tmnt.sparse.config import get_default_cfg
    from tmnt.sparse.modeling import BatchTopKSAE
    cfg = get_default_cfg()
    cfg['activation_path'] = '/Users/wellner/Projects/SIREN2025/t1.arrow'
    cfg['device'] = 'cpu'
    cfg['num_samples'] = 20000
    cfg['batch_size'] = 200
    act_store = ActivationsStore(cfg)
    sed = BatchTopKSAE(cfg)
    train_sparse_encoder_decoder(sed, act_store, cfg)
    return sed

    

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import tqdm
from datasets.arrow_writer import ArrowWriter
from tmnt.inference import SeqVEDInferencer
import io, json


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

def build_activation_store(json_input_texts, emb_model_path, arrow_output):

    inf_model = SeqVEDInferencer.from_saved(emb_model_path, device='cuda')
    with io.open(json_input_texts) as fp:
        with ArrowWriter(path=arrow_output) as writer:
            for l in fp:
                js = json.loads(l)
                enc = inf_model.get_text_embedding(js['text'])
                writer.write({'data': enc})
            writer.finalize()


def train_sparse_encoder_decoder(sed, activation_store, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sed.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    for i in pbar:
        batch = activation_store.next_batch()
        sed_output = sed(batch)

        loss = sed_output["loss"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sed_output['l0_norm']:.4f}", "L2": f"{sed_output['l2_loss']:.4f}", "L1": f"{sed_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sed.parameters(), cfg["max_grad_norm"])
        sed.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    

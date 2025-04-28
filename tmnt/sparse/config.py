import torch 

__all__ = ['get_default_cfg']

def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_samples": int(1e9),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "dtype": torch.float32,
        "site": "resid_pre",
        "layer": 8,
        "act_size": 768,
        "dict_size": 12288,
        "device": "cuda:0",
        "input_unit_norm": True,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 5,

        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        # for jumprelu
        "bandwidth": 0.001,
    }
    return default_cfg
import torch


BASE_SETTINGS = {
    "obs_dim": 8,
    "num_processes": 8,
    "num_tasks": 4,
    "num_layers": 3,
    "split_step_threshold": 30,
    "sharing_threshold": 0.5,
    "cap_sample_size": True,
    "ema_alpha": 0.999,
    "device": torch.device("cpu"),
}
V1_SETTINGS = {**BASE_SETTINGS, "split_alpha": 0.05, "grad_var": None}
V2_SETTINGS = {**BASE_SETTINGS, "split_freq": 4, "splits_per_step": 3}

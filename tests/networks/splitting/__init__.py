import torch


SETTINGS = {
    "obs_dim": 8,
    "num_processes": 8,
    "num_tasks": 4,
    "num_layers": 3,
    "split_alpha": 0.05,
    "grad_var": None,
    "split_step_threshold": 30,
    "cap_sample_size": True,
    "ema_alpha": 0.999,
    "include_task_index": True,
    "device": torch.device("cpu"),
}

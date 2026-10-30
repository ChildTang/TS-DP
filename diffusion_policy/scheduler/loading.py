"""Builds a TS-DP policy and a scheduler env from checkpoints."""
import os
from typing import List, Optional

import dill
import hydra
import torch

from diffusion_policy.scheduler.env import SchedulerEnv
from diffusion_policy.scheduler.tasks import TaskSpec


def load_ts_dp_policy(checkpoint: str, drafter_checkpoint: str, device: str):
    """Target model from a Diffusion Policy checkpoint (the official ones) plus
    the released drafter of that checkpoint."""
    with open(checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    policy = hydra.utils.instantiate(cfg.policy, use_speculative_decoding=True)
    key = 'ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][key])
    policy.load_draft_checkpoint(drafter_checkpoint)
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def make_scheduler_env_fn(task: TaskSpec,
        checkpoint: str,
        drafter_checkpoint: str,
        device: str,
        output_dir: str,
        task_indices: List[int],
        update_interval: Optional[int] = None,
        verbose: bool = False):
    """Env factory for tianshou vector envs (each worker loads its own policy)."""
    def _init():
        policy, cfg = load_ts_dp_policy(checkpoint, drafter_checkpoint, device)
        os.makedirs(output_dir, exist_ok=True)
        # a single simulator is enough: the scheduler env only uses the first one
        base_runner = hydra.utils.instantiate(cfg.task.env_runner,
            output_dir=output_dir, n_envs=1, n_train_vis=0, n_test_vis=0)
        return SchedulerEnv(task, base_runner, policy,
            task_indices=task_indices,
            update_interval=update_interval,
            verbose=verbose)
    return _init

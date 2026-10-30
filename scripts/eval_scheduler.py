"""Evaluate TS-DP with a trained PPO scheduler on all initial conditions of a task.

Every initial condition of the Diffusion Policy env runner (n_train dataset
states + n_test random seeds) is run once. Results are written to
<output-dir>/eval_<task>.json.

Usage:
python scripts/eval_scheduler.py --task can \
    --checkpoint data/checkpoints/image/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
    --drafter-checkpoint data/ts_dp/can/drafter.ckpt \
    --ppo-checkpoint data/ts_dp/can/scheduler.pth \
    --output-dir data/eval/can --device cuda:0
"""
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

import argparse
import json
import multiprocessing
import os
import shutil

import dill
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv

from diffusion_policy.scheduler.loading import make_scheduler_env_fn
from diffusion_policy.scheduler.ppo import build_ppo_policy
from diffusion_policy.scheduler.tasks import TASKS, get_task

RESULT_KEYS = ['task_success', 'task_max_reward', 'task_total_reward', 'n_completed_tasks',
    'p_1', 'p_2', 'p_3', 'p_4', 'avg_success',
    'spec/accepted_drafts', 'spec/total_drafts', 'spec/accept_rate', 'spec/inference_count',
    'task_duration']
STEP_KEYS = ['current_step', 'param/sigma', 'param/threshold', 'param/draft_high',
    'param/draft_medium', 'param/draft_low', 'spec/accept_rate']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=sorted(TASKS))
    parser.add_argument('--checkpoint', required=True, help='Diffusion Policy checkpoint (target model)')
    parser.add_argument('--drafter-checkpoint', required=True)
    parser.add_argument('--ppo-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-parallel-envs', type=int, default=28,
        help='initial conditions evaluated in parallel (one policy copy each)')
    parser.add_argument('--task-indices', type=int, nargs='+', default=None,
        help='subset of initial conditions (default: all)')
    parser.add_argument('--update-interval', type=int, default=None,
        help='env steps between scheduler decisions (default: task setting)')
    parser.add_argument('--verbose', action='store_true', help='print every scheduler decision')
    return parser.parse_args()


def episode_results(buffer):
    """One entry per finished episode: final metrics and the parameter trajectory."""
    results = []
    for sub_buffer in buffer.buffers:
        trajectory = []
        for i in sub_buffer.sample_indices(0):
            info = sub_buffer.info[i]
            trajectory.append({k: float(info[k]) for k in STEP_KEYS})
            if bool(info['task_done']):
                result = {'task_id': int(info['task_id'])}
                result.update({k: float(info[k]) for k in RESULT_KEYS if k in info})
                result['trajectory'] = trajectory
                results.append(result)
                trajectory = []
    return results


def summarize(results):
    if not results:
        return {}
    summary = {'n_episodes': len(results)}
    for k in RESULT_KEYS:
        values = [r[k] for r in results if k in r]
        if values:
            summary[k] = float(np.mean(values))
    n_infer = summary.get('spec/inference_count', 0)
    if n_infer > 0:
        summary['spec/accepted_drafts_per_inference'] = summary['spec/accepted_drafts'] / n_infer
    return summary


def main():
    args = parse_args()
    task = get_task(args.task)
    update_interval = args.update_interval or task.eval_update_interval or task.update_interval
    os.makedirs(args.output_dir, exist_ok=True)
    runner_dir = os.path.join(args.output_dir, 'runner_cache')

    with open(args.checkpoint, 'rb') as f:
        runner_cfg = torch.load(f, pickle_module=dill, map_location='cpu')['cfg'].task.env_runner
    n_train, n_test = int(runner_cfg.n_train), int(runner_cfg.n_test)
    task_indices = args.task_indices or list(range(n_train + n_test))

    ppo_state_dict = torch.load(args.ppo_checkpoint, map_location=args.device)
    policy = None
    results = []
    for start in range(0, len(task_indices), args.n_parallel_envs):
        chunk = task_indices[start:start + args.n_parallel_envs]
        print(f"Evaluating initial conditions {chunk[0]}..{chunk[-1]} ({len(chunk)} envs)", flush=True)
        envs = SubprocVectorEnv([make_scheduler_env_fn(task,
                checkpoint=args.checkpoint,
                drafter_checkpoint=args.drafter_checkpoint,
                device=args.device,
                output_dir=os.path.join(runner_dir, f'worker_{i}'),
                task_indices=[idx],
                update_interval=update_interval,
                verbose=args.verbose)
            for i, idx in enumerate(chunk)])
        if policy is None:
            policy = build_ppo_policy(task, envs.observation_space[0], envs.action_space[0],
                device=args.device)
            policy.load_state_dict(ppo_state_dict)
            policy.eval()
        collector = Collector(policy, envs, VectorReplayBuffer(task.max_steps * len(chunk), len(chunk)))
        collector.collect(n_episode=len(chunk))
        results += episode_results(collector.buffer)
        envs.close()
        torch.cuda.empty_cache()

    for r in results:
        r['split'] = 'train' if r['task_id'] < n_train else 'test'
    output = {
        'task': args.task,
        'checkpoint': args.checkpoint,
        'drafter_checkpoint': args.drafter_checkpoint,
        'ppo_checkpoint': args.ppo_checkpoint,
        'update_interval': update_interval,
        'summary': {
            'all': summarize(results),
            'train': summarize([r for r in results if r['split'] == 'train']),
            'test': summarize([r for r in results if r['split'] == 'test']),
        },
        'episodes': sorted(results, key=lambda r: r['task_id']),
    }
    path = os.path.join(args.output_dir, f'eval_{args.task}.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    shutil.rmtree(runner_dir, ignore_errors=True)
    print(json.dumps(output['summary'], indent=2))
    print(f"Saved to {path}")


if __name__ == '__main__':
    # each env worker runs its own policy on the GPU
    multiprocessing.set_start_method('spawn', force=True)
    main()

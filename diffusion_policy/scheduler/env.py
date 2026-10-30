"""Gym environment in which the PPO scheduler sets the speculative decoding
parameters of a TS-DP policy while it executes manipulation episodes.

One gym step = ``update_interval`` environment steps executed with a fixed
parameter set. The env runs the given initial conditions one after another and
reports the task metrics in ``info`` when an episode ends.
"""
import time
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.scheduler.tasks import TaskSpec

# metrics reported at the end of an episode, per outcome type
OUTCOME_KEYS = {
    'binary': ['task_success'],
    'pusht': ['task_success', 'task_max_reward'],
    'kitchen': ['task_success', 'n_completed_tasks', 'p_1', 'p_2', 'p_3', 'p_4', 'avg_success'],
    'blockpush': ['task_success', 'n_completed_tasks', 'p_1', 'p_2', 'avg_success',
        'task_total_reward'],
}


class SchedulerEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self,
            task: TaskSpec,
            base_runner,
            policy,
            task_indices: List[int],
            update_interval: Optional[int] = None,
            verbose: bool = False):
        """
        base_runner: env runner of the Diffusion Policy task (provides the vectorized
            env and the initial conditions; only its first env is used).
        policy: TS-DP policy with speculative decoding enabled.
        task_indices: initial conditions (indices into base_runner.env_init_fn_dills),
            run in this order.
        """
        super().__init__()
        if not getattr(policy, 'use_speculative_decoding', False):
            raise ValueError("The policy must be created with use_speculative_decoding=True.")
        self.task = task
        self.base_runner = base_runner
        self.policy = policy
        self.task_indices = list(task_indices)
        self.update_interval = update_interval or task.update_interval
        self.max_steps = task.max_steps
        self.verbose = verbose
        self.render_mode = None

        self.action_dims = task.action_dims
        self.action_space = gym.spaces.Discrete(int(np.prod(self.action_dims)))
        self.observation_space = self._make_observation_space()

        self._episode_idx = 0
        self._obs = None
        self._info = None
        self._past_action = None
        self._step = 0
        self._accepted = 0
        self._proposed = 0
        self._n_inference = 0
        self._start_time = 0.0

    # ========= gym API ============
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._start_episode(0)
        return self._get_observation(), self._make_info(outcome=None, action=-1)

    def step(self, action: int):
        action = int(action)
        self._apply_parameters(action)
        if self.verbose:
            p = self.policy
            print(f"[task {self.task_indices[self._episode_idx]} step {self._step:3d}] "
                  f"sigma={p.sigma_scale:.2f} thr={p.accept_threshold:.2f} "
                  f"draft=[H:{p.draft_steps['high_steps']}, M:{p.draft_steps['medium_steps']}, "
                  f"L:{p.draft_steps['low_steps']}]", flush=True)

        env_reward = 0.0
        done = False
        target_step = min(self._step + self.update_interval, self.max_steps)
        while self._step < target_step:
            reward, done = self._env_step()
            env_reward += reward
            if done or self._step >= self.max_steps:
                done = True
                break

        outcome = self._episode_outcome() if done else None
        info = self._make_info(outcome, action)

        terminated = done
        if done and self._episode_idx + 1 < len(self.task_indices):
            # continue with the next initial condition
            self._start_episode(self._episode_idx + 1)
            terminated = False
        return self._get_observation(), float(env_reward), terminated, False, info

    def close(self):
        env = getattr(self.base_runner, 'env', None)
        if env is not None:
            env.close()

    # ========= episode ============
    def _start_episode(self, episode_idx: int):
        self._episode_idx = episode_idx
        runner = self.base_runner
        init_fn = runner.env_init_fn_dills[self.task_indices[episode_idx]]
        n_envs = len(runner.env_fns)
        runner.env.call_each('run_dill_function',
            args_list=[(init_fn,)] + [(runner.env_init_fn_dills[0],)] * (n_envs - 1))
        self._obs = runner.env.reset()
        self._info = None
        self._past_action = None
        self._step = 0
        self._accepted = 0
        self._proposed = 0
        self._n_inference = 0
        self._start_time = time.time()
        self.policy.reset()

    def _env_step(self):
        runner = self.base_runner
        if isinstance(self._obs, dict):
            np_obs_dict = {k: np.array(v, copy=True) for k, v in self._obs.items()}
        else:
            np_obs_dict = {'obs': np.array(self._obs, copy=True)}
        if getattr(runner, 'past_action', False) and self._past_action is not None:
            np_obs_dict['past_action'] = self._past_action[
                :, -(runner.n_obs_steps - 1):].astype(np.float32)
        obs_dict = dict_apply(np_obs_dict,
            lambda x: torch.from_numpy(x).to(device=self.policy.device))

        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        self._n_inference += 1
        self._accepted += int(self.policy.accepted_drafts)
        self._proposed += int(self.policy.proposed_drafts)

        pred_action = action_dict['action'].detach().to('cpu').numpy()
        if not np.all(np.isfinite(pred_action)):
            raise RuntimeError("Nan or Inf action")
        env_action = pred_action
        if getattr(runner, 'abs_action', False):
            env_action = runner.undo_transform_action(pred_action)

        self._obs, reward, done, info = runner.env.step(env_action)
        self._info = info
        self._past_action = pred_action
        self._step += runner.n_action_steps
        return float(reward[0]), bool(done[0])

    def _episode_outcome(self) -> Dict[str, float]:
        """Task metrics of the finished episode."""
        kind = self.task.outcome
        rewards = self.base_runner.env.call('get_attr', 'reward')[0]
        rewards = np.asarray(rewards if rewards is not None else [], dtype=np.float64)
        if kind == 'binary':
            max_reward = float(rewards.max()) if rewards.size else 0.0
            return {'task_success': float(max_reward >= 0.9)}
        if kind == 'pusht':
            max_reward = float(rewards.max()) if rewards.size else 0.0
            return {'task_success': float(max_reward >= 0.9), 'task_max_reward': max_reward}
        if kind == 'kitchen':
            # MultiStepWrapper stacks info over time; the last entry is the current set
            n_completed = len(self._info[0]['completed_tasks'][-1]) if self._info else 0
            metrics = {f'p_{n}': float(n_completed >= n) for n in range(1, 5)}
            metrics.update(n_completed_tasks=float(n_completed),
                task_success=float(n_completed >= 4),
                avg_success=float(np.mean([metrics[f'p_{n}'] for n in range(1, 5)])))
            return metrics
        if kind == 'blockpush':
            total_reward = float(np.unique(rewards).sum()) if rewards.size else 0.0
            n_completed = 2 if total_reward > 0.9 else (1 if total_reward > 0.4 else 0)
            metrics = {'p_1': float(n_completed >= 1), 'p_2': float(n_completed >= 2)}
            metrics.update(n_completed_tasks=float(n_completed),
                task_success=float(n_completed >= 2), task_total_reward=total_reward,
                avg_success=(metrics['p_1'] + metrics['p_2']) / 2)
            return metrics
        raise ValueError(f"Unknown outcome type '{kind}'")

    def _make_info(self, outcome, action) -> Dict:
        p = self.policy
        info = {
            'task_id': int(self.task_indices[self._episode_idx]),
            'current_step': int(self._step),
            'task_done': outcome is not None,
            'param/action': int(action),
            'param/sigma': float(p.sigma_scale),
            'param/threshold': float(p.accept_threshold),
            'param/draft_high': int(p.draft_steps['high_steps']),
            'param/draft_medium': int(p.draft_steps['medium_steps']),
            'param/draft_low': int(p.draft_steps['low_steps']),
            'spec/accepted_drafts': int(self._accepted),
            'spec/total_drafts': int(self._proposed),
            'spec/accept_rate': float(self._accepted / self._proposed) if self._proposed else 0.0,
            'spec/inference_count': int(self._n_inference),
            'task_duration': float(time.time() - self._start_time) if outcome is not None else 0.0,
        }
        if outcome is None:
            # keep the info keys identical at every step (tianshou stacks them)
            outcome = {k: 0.0 for k in OUTCOME_KEYS[self.task.outcome]}
        info.update(outcome)
        return info

    # ========= observation / action ============
    def _make_observation_space(self):
        t = self.task
        # per-env space of the base task, observations stacked over n_obs_steps
        env_space = self.base_runner.env.single_observation_space
        if t.obs_type == 'robomimic':
            spaces = {k: gym.spaces.Box(0, 1, shape=tuple(env_space[k].shape[-3:]), dtype=np.float32)
                for k in t.image_keys}
            spaces['state'] = gym.spaces.Box(-1, 1, shape=(9 * len(t.robots) + 1,), dtype=np.float32)
            return gym.spaces.Dict(spaces)
        if t.obs_type == 'pusht':
            return gym.spaces.Dict({
                'image': gym.spaces.Box(0, 1, shape=(3, 96, 96), dtype=np.float32),
                'state': gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32)})
        if t.obs_type == 'lowdim':
            return gym.spaces.Box(-8, 8, shape=(env_space.shape[-1],), dtype=np.float32)
        raise ValueError(f"Unknown obs_type '{t.obs_type}'")

    def _get_observation(self):
        """Latest frame of the first env, plus episode progress."""
        t = self.task
        obs = self._obs
        progress = self._step / self.max_steps
        if t.obs_type == 'robomimic':
            out = {}
            for k in t.image_keys:
                img = obs[k][0, -1].astype(np.float32) / 255.0
                out[k] = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            state = []
            for r in t.robots:
                state += [obs[f'robot{r}_eef_pos'][0, -1], obs[f'robot{r}_eef_quat'][0, -1],
                          obs[f'robot{r}_gripper_qpos'][0, -1]]
            state = np.concatenate(state + [[progress]]).astype(np.float32)
            out['state'] = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            return out
        if t.obs_type == 'pusht':
            img = np.asarray(obs['image'][0, -1], dtype=np.float32)
            state = np.zeros(3, dtype=np.float32)
            state[:2] = np.asarray(obs['agent_pos'][0, -1], dtype=np.float32) / 512.0
            state[2] = progress
            return {'image': np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0),
                    'state': np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)}
        vec = obs['obs'] if isinstance(obs, dict) else obs
        vec = np.nan_to_num(vec[0, -1], nan=0.0, posinf=8.0, neginf=-8.0)
        return vec.astype(np.float32)

    def decode_action(self, action: int) -> List[int]:
        """Flat action index -> [sigma, threshold, high, medium, low] option indices."""
        idx = [0] * len(self.action_dims)
        for i in range(len(self.action_dims) - 1, -1, -1):
            idx[i] = action % self.action_dims[i]
            action //= self.action_dims[i]
        return idx

    def _apply_parameters(self, action: int):
        t = self.task
        s, th, hi, md, lo = self.decode_action(action)
        self.policy.sigma_scale = float(t.sigma_options[s])
        self.policy.accept_threshold = float(t.threshold_options[th])
        self.policy.draft_steps = {
            'high_steps': int(t.high_steps_options[hi]),
            'medium_steps': int(t.medium_steps_options[md]),
            'low_steps': int(t.low_steps_options[lo]),
        }

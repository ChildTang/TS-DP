"""Speculative decoding for transformer-based Diffusion Policy (TS-DP).

At every target-model denoising step, the drafter proposes several future
denoising steps autoregressively. The target model then scores all proposals in
one batched forward pass and accepts the longest prefix that passes a
Metropolis-Hastings style test; the first rejected draft is replaced by a
reflection-coupled sample from the target posterior.

The three quantities that the PPO scheduler adjusts at run time are
``sigma_scale``, ``accept_threshold`` and ``draft_steps`` (draft budget in the
high / medium / low noise stages).
"""
from typing import Dict, List, Optional, Tuple

import dill
import torch

from diffusion_policy.model.diffusion.ddpm_utils import (
    DDPMStepOutput, ddpm_step_with_stats, ddpm_posterior_mean)
from diffusion_policy.model.diffusion.transformer_for_diffusion import DraftTransformerForDiffusion

DRAFT_PREFIX = 'draft_model.'


class SpeculativeDecodingMixin:
    """Adds a drafter and speculative sampling to a diffusion policy.

    The host class must define ``self.model`` (TransformerForDiffusion),
    ``self.noise_scheduler`` and ``self.num_inference_steps``.
    """

    # Stage boundaries for the draft budget, in remaining denoising steps.
    HIGH_NOISE_REMAINING_STEPS = 90
    LOW_NOISE_REMAINING_STEPS = 10

    def _init_speculative_decoding(self,
            use_speculative_decoding: bool,
            draft_checkpoint_path: Optional[str],
            sigma_scale: float,
            accept_threshold: float,
            draft_steps: Dict[str, int]):
        self.use_speculative_decoding = use_speculative_decoding
        self.draft_model = None
        if use_speculative_decoding:
            self.draft_model = DraftTransformerForDiffusion.from_target(self.model)
            if draft_checkpoint_path is not None:
                self.load_draft_checkpoint(draft_checkpoint_path)

        # run-time parameters (set by the PPO scheduler)
        self.sigma_scale = sigma_scale
        self.accept_threshold = accept_threshold
        self.draft_steps = dict(draft_steps)

        # statistics of the last call to predict_action
        self.accepted_drafts = 0
        self.proposed_drafts = 0

    # ========= checkpoints ============
    def load_state_dict(self, state_dict, strict=True):
        """Target-model checkpoints (e.g. the official Diffusion Policy ones) contain
        no drafter weights; only those keys are allowed to be missing."""
        if self.draft_model is None or any(k.startswith(DRAFT_PREFIX) for k in state_dict):
            return super().load_state_dict(state_dict, strict=strict)
        result = super().load_state_dict(state_dict, strict=False)
        if strict:
            missing = [k for k in result.missing_keys if not k.startswith(DRAFT_PREFIX)]
            if missing or result.unexpected_keys:
                raise RuntimeError(
                    f"Error loading state_dict: missing keys {missing}, "
                    f"unexpected keys {result.unexpected_keys}")
        return result

    def load_draft_checkpoint(self, checkpoint_path: str):
        """Load drafter weights from a released drafter checkpoint."""
        if self.draft_model is None:
            raise RuntimeError("use_speculative_decoding=False: the policy has no drafter.")
        with open(checkpoint_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill, map_location='cpu')
        state_dict = payload['state_dicts']['model']
        draft_state_dict = {
            k[len(DRAFT_PREFIX):]: v for k, v in state_dict.items() if k.startswith(DRAFT_PREFIX)}
        if not draft_state_dict:
            raise KeyError(f"No drafter weights ('{DRAFT_PREFIX}*') found in {checkpoint_path}")
        self.draft_model.load_state_dict(draft_state_dict)

    # ========= inference ============
    def _draft_budget(self, remaining_steps: int) -> int:
        if remaining_steps > self.HIGH_NOISE_REMAINING_STEPS:
            return self.draft_steps['high_steps']
        if remaining_steps > self.LOW_NOISE_REMAINING_STEPS:
            return self.draft_steps['medium_steps']
        return self.draft_steps['low_steps']

    def conditional_sample_speculative(self,
            condition_data, condition_mask,
            cond=None, generator=None, **kwargs):
        """Speculative counterpart of ``conditional_sample``.

        kwargs are forwarded to ``scheduler.step`` for the target model steps.
        """
        if condition_data.shape[0] != 1:
            raise ValueError("Speculative decoding supports batch size 1 only.")
        model = self.model
        draft_model = self.draft_model
        scheduler = self.noise_scheduler
        self.accepted_drafts = 0
        self.proposed_drafts = 0

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        n_steps = len(timesteps)
        step_idx = 0
        while step_idx < n_steps:
            t = timesteps[step_idx]
            trajectory[condition_mask] = condition_data[condition_mask]

            # one target-model step: x_t -> x_{t-1}
            target_output = model(trajectory, t, cond)
            next_trajectory = scheduler.step(
                target_output, t, trajectory,
                generator=generator, **kwargs).prev_sample
            next_trajectory[condition_mask] = condition_data[condition_mask]

            # drafter proposes the following steps
            drafts: List[torch.Tensor] = []
            draft_stats: List[DDPMStepOutput] = []
            if step_idx >= 1:
                current = next_trajectory
                for k in range(self._draft_budget(n_steps - step_idx)):
                    if step_idx + k + 1 >= n_steps:
                        break
                    future_t = timesteps[step_idx + k + 1]
                    if future_t == 0:
                        break
                    draft_output = draft_model(current, future_t, cond)
                    stats = ddpm_step_with_stats(
                        scheduler, draft_output, future_t, current, generator=generator)
                    draft = stats.prev_sample
                    draft[condition_mask] = condition_data[condition_mask]
                    drafts.append(draft)
                    draft_stats.append(stats)
                    current = draft
                self.proposed_drafts += len(drafts)

            if len(drafts) > 0:
                n_accepted = self._verify_drafts(
                    next_trajectory, drafts, draft_stats,
                    timesteps[step_idx + 1: step_idx + 1 + len(drafts)], cond)
                self.accepted_drafts += n_accepted
                trajectory = drafts[n_accepted - 1]
                step_idx += n_accepted + 1
            else:
                trajectory = next_trajectory
                step_idx += 1

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    @torch.no_grad()
    def _verify_drafts(self,
            next_trajectory: torch.Tensor,
            drafts: List[torch.Tensor],
            draft_stats: List[DDPMStepOutput],
            draft_timesteps: torch.Tensor,
            cond: Optional[torch.Tensor]) -> int:
        """Score all drafts with the target model in one batch.

        Draft k was produced from input ``inputs[k]`` at timestep ``draft_timesteps[k]``,
        where ``inputs = [next_trajectory] + drafts[:-1]``.
        Returns the number of accepted drafts; if a draft is rejected it is replaced
        in-place by a corrected sample and counted as accepted.
        """
        n_drafts = len(drafts)
        device = drafts[0].device
        inputs = torch.stack([next_trajectory] + drafts[:-1], dim=0)  # (K,B,T,D)
        timesteps = torch.stack(list(draft_timesteps)).to(device)    # (K,)

        K, B = inputs.shape[:2]
        batch_cond = cond.repeat(K, 1, 1) if cond is not None else None
        target_outputs = self.model(
            inputs.reshape(K * B, *inputs.shape[2:]),
            timesteps.repeat_interleave(B),
            batch_cond)
        target_outputs = target_outputs.view(K, B, *target_outputs.shape[1:])

        target_means = ddpm_posterior_mean(
            self.noise_scheduler, target_outputs, timesteps, inputs)
        draft_means = torch.stack([s.pred_mean for s in draft_stats], dim=0).to(device)
        variances = torch.stack([s.variance for s in draft_stats], dim=0).to(device)
        noises = torch.stack([s.variance_noise for s in draft_stats], dim=0).to(device)

        # batch size is 1: (K,1,T,D) -> (K,T,D)
        first_rejected, corrected = self._accept_drafts(
            noises.squeeze(1), draft_means.squeeze(1), target_means.squeeze(1), variances)
        if first_rejected is None:
            return n_drafts
        drafts[first_rejected] = corrected
        return first_rejected + 1

    def _accept_drafts(self,
            xi: torch.Tensor,
            draft_mean: torch.Tensor,
            target_mean: torch.Tensor,
            variance: torch.Tensor) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Acceptance test for K drafts.

        xi: (K,T,D) noise used to sample each draft; draft_mean/target_mean: (K,T,D);
        variance: (K,) posterior variance.
        Returns (index of first rejected draft or None, corrected sample (1,T,D) or None).
        """
        K = xi.shape[0]
        v = draft_mean - target_mean
        small_diff = v.abs().view(K, -1).max(dim=1)[0] < 1e-6

        sigma = self.sigma_scale + 0.5 * torch.sigmoid(torch.sqrt(variance))
        sigma = sigma.view(K, 1, 1)
        delta = v / sigma
        v_norm = torch.sqrt(torch.sum(v.view(K, -1) ** 2, dim=1, keepdim=True))
        e = v / v_norm.view(K, 1, 1)

        delta_norm2 = torch.sum(delta.view(K, -1) ** 2, dim=1)
        delta_xi = torch.sum((delta * xi).view(K, -1), dim=1)
        accept_prob = torch.exp(-(0.5 * delta_norm2 + delta_xi))
        accepted = (self.accept_threshold <= torch.minimum(torch.ones_like(accept_prob), accept_prob)) | small_diff

        rejected = ~accepted
        if not rejected.any():
            return None, None
        i = int(rejected.nonzero(as_tuple=True)[0][0])

        # reflection coupling: sample from the target posterior
        xi_i = xi[i:i+1]
        e_i = e[i:i+1]
        e_xi = torch.sum(e_i * xi_i, dim=(-2, -1), keepdim=True)
        corrected = target_mean[i:i+1] + torch.sqrt(variance[i:i+1]) * (xi_i - 2 * e_xi * e_i)
        return i, corrected

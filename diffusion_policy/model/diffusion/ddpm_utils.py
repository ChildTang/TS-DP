"""DDPM helpers used by speculative decoding.

These functions operate on a stock ``diffusers.DDPMScheduler`` (tested with
diffusers 0.11.1), so no patching of the installed ``diffusers`` package is
required.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


@dataclass
class DDPMStepOutput:
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    # mean of p(x_{t-1} | x_t)
    pred_mean: torch.Tensor
    # variance used to sample x_{t-1} (std if variance_type == "fixed_small_log"); None at t == 0
    variance: Optional[torch.Tensor]
    # standard Gaussian noise used to sample x_{t-1}; None at t == 0
    variance_noise: Optional[torch.Tensor]


def _pred_original_sample(scheduler: DDPMScheduler, model_output, sample, alpha_prod_t, beta_prod_t):
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        pred = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    elif prediction_type == "sample":
        pred = model_output
    elif prediction_type == "v_prediction":
        pred = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
    else:
        raise ValueError(f"Unsupported prediction type {prediction_type}")
    if scheduler.config.clip_sample:
        pred = torch.clamp(pred, -1, 1)
    return pred


def ddpm_step_with_stats(
        scheduler: DDPMScheduler,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None) -> DDPMStepOutput:
    """Equivalent to ``DDPMScheduler.step`` (same arithmetic and RNG consumption),
    but additionally returns the posterior mean, the variance and the noise sample,
    which are needed to verify drafts.
    """
    t = timestep
    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    pred_original_sample = _pred_original_sample(
        scheduler, model_output, sample, alpha_prod_t, beta_prod_t)

    # formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * scheduler.betas[t]) / beta_prod_t
    current_sample_coeff = scheduler.alphas[t] ** 0.5 * beta_prod_t_prev / beta_prod_t
    pred_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    prev_sample = pred_mean
    variance = None
    variance_noise = None
    if t > 0:
        variance_noise = torch.randn(
            model_output.shape, generator=generator,
            device=model_output.device, dtype=model_output.dtype)
        variance = scheduler._get_variance(t, predicted_variance=predicted_variance)
        if scheduler.variance_type == "fixed_small_log":
            prev_sample = prev_sample + variance * variance_noise
        else:
            prev_sample = prev_sample + (variance ** 0.5) * variance_noise

    return DDPMStepOutput(
        prev_sample=prev_sample,
        pred_original_sample=pred_original_sample,
        pred_mean=pred_mean,
        variance=variance,
        variance_noise=variance_noise)


def ddpm_posterior_mean(
        scheduler: DDPMScheduler,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor) -> torch.Tensor:
    """Batched mean of p(x_{t-1} | x_t).

    model_output, sample: (N, ...); timesteps: (N,), one timestep per leading index.
    """
    device = model_output.device
    n = model_output.shape[0]
    view_shape = (n,) + (1,) * (model_output.ndim - 1)
    timesteps = timesteps.cpu()

    alpha_prod_t = scheduler.alphas_cumprod[timesteps].to(device).view(view_shape)
    alpha_prod_t_prev = torch.where(
        timesteps > 0,
        scheduler.alphas_cumprod[timesteps - 1],
        scheduler.one
    ).to(device).view(view_shape)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    pred_original_sample = _pred_original_sample(
        scheduler, model_output, sample, alpha_prod_t, beta_prod_t)

    betas_t = scheduler.betas[timesteps].to(device).view(view_shape)
    alphas_t = scheduler.alphas[timesteps].to(device).view(view_shape)
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * betas_t) / beta_prod_t
    current_sample_coeff = alphas_t ** 0.5 * beta_prod_t_prev / beta_prod_t
    return pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

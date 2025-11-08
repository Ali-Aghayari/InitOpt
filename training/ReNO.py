import json
import logging
import os

import torch
from datasets import load_dataset
from pytorch_lightning import seed_everything
from tqdm import tqdm

from Rtraining import LatentNoiseTrainer, get_optimizer


def apply_reno(prompt, seed, save_dir, reward_losses, pipe):
    
    # seed_everything(seed)

    device = torch.device("cuda")

    dtype = torch.float16

    
    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=50,
        n_inference_steps=1,
        seed=seed,
        save_all_images=False,
        device=device,
        no_optim=False,
        regularize=True,
        regularization_weight=0.01,
        grad_clip=0.1,
        log_metrics=False,
        imageselect=False,
    )

    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    shape = (
        1,
        pipe.unet.in_channels,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor,
    )

    enable_grad = True
    multi_apply_fn = None

    init_latents = torch.randn(shape, device=device, dtype=dtype)
    latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
    optimizer = get_optimizer("sgd", latents, 5, True)
    save_dir = f"{save_dir}/{prompt[:150]}"
    init_image, best_image, inti_latent, best_latent, init_SD_latents, best_SD_latents = trainer.train(
        latents, prompt, optimizer, save_dir, multi_apply_fn
    )
    
    # os.makedirs(f"{save_dir}", exist_ok=True)
    # best_image.save(f"{save_dir}/best_image.png")
    # init_image.save(f"{save_dir}/init_image.png")

    return init_image, best_image, inti_latent, best_latent, init_SD_latents, best_SD_latents
    
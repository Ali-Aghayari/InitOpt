import json
import os
import math
import random
import argparse
import numpy as np
import accelerate
from torch import nn
import copy

import torch
import einops
from torch.nn.functional import mse_loss
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler
from reward_model.eval_pickscore import PickScore
from typing  import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import accelerate
from model import NoiseUnet
from reward_model.eval_pickscore import PickScore
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.models.normalization import AdaGroupNorm


from diffusers import DiffusionPipeline


__all__ = ['Embedding_Solver']

class Embedding_Solver:
    def __init__(
            self,
            pipeline: nn.Module,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            config=None,
            local_rank=None,

    ):
        self.config = config
        self.pipeline = pipeline

        self.conv_in = copy.deepcopy(self.pipeline.unet.conv_in)
        self.conv_in.load_state_dict(self.pipeline.unet.conv_in.state_dict().copy())
        self.out_channels = self.conv_in.out_channels
        self.in_channels = self.conv_in.in_channels

        from model import NoiseTransformer
        self.unet = NoiseTransformer().to(device).to(torch.float32)
        self.text_embedding = AdaGroupNorm(1024 * 77, 4, 1, eps=1e-6).to(device).to(torch.float32)

        self.optimizer = torch.optim.AdamW(list(self.unet.parameters()) + list(self.text_embedding.parameters()),
                                           lr=1e-4)

        self.local_rank = local_rank
        self.device = device
        self.init()

    def init(self):
        self.pipeline.to(self.device)
        self.unet.to(self.device)
        self.unet.train()

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            total_epoch=5,
            save_path='./golden_unet'
    ):
        for epoch in range(1, total_epoch + 1):
            self.unet.train()
            self.text_embedding.train()
            train_loss, count = 0, 0

            # train
            pbar = tqdm(train_loader)
            for step, (original_noise, optimized_noise, prompt) in enumerate(pbar, 1):
                original_noise, optimized_noise = original_noise.to(self.device), optimized_noise.to(self.device)

                (
                    prompt_embeds,
                    _
                ) = self.pipeline.encode_prompt(prompt=prompt[0], device=self.device, do_classifier_free_guidance=True, num_images_per_prompt=1)

                prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)

                difference = (original_noise.float() - optimized_noise.float()).detach()

                text_emb = self.text_embedding(original_noise.float(), prompt_embeds)
                encoder_hidden_states = original_noise + text_emb

                golden_embedding = self.unet(encoder_hidden_states.float())

                loss = mse_loss(golden_embedding, difference.float())
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                if step % 50 == 0:
                    pbar.set_postfix_str(f"loss={train_loss / 50}")
                    train_loss = 0.

            with torch.no_grad():
                self.unet.eval()
                self.text_embedding.eval()

                total_eval_loss = 0.
                count = 0.

                for i, (original_noise, optimized_noise, prompt) in enumerate(val_loader):
                    original_noise, optimized_noise = original_noise.to(self.device), optimized_noise.to(self.device)

                    (
                        prompt_embeds,
                        _
                    ) = self.pipeline.encode_prompt(prompt=prompt[0], device=self.device, do_classifier_free_guidance=True, num_images_per_prompt=1)

                    prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
                    text_emb = self.text_embedding(original_noise.float(), prompt_embeds)
                    encoder_hidden_states = original_noise + text_emb

                    difference = (original_noise.float() - optimized_noise.float()).detach()

                    golden_embedding = self.unet(encoder_hidden_states.float())

                    total_eval_loss += mse_loss(golden_embedding, difference) * len(original_noise)
                    count += len(original_noise)

                print("Eval Loss:", round(total_eval_loss.item() * 100 / count, 2), "%")

            # self.scheduler.step(loss, epoch)

        torch.save({"unet": self.unet.state_dict(), "embeeding": self.text_embedding.state_dict()}, f"{save_path}.pth")

        return self.unet

    def generate(self,
                 latent,
                 optimized=None,
                 reward_model=None,
                 prompt=None,
                 save_postfix=None,
                 save_pic=None,
                 idx=None,
                 config=None,
                 preprocessor=None,
                 image_processor=None,
                 tokenizer=None
    ):

        (
            prompt_embeds,
            _
        ) = self.pipeline.encode_prompt(prompt=prompt, device=self.device, do_classifier_free_guidance=True, num_images_per_prompt=1)
        
        prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
        text_emb = self.text_embedding(latent.float(), prompt_embeds)
        encoder_hidden_states = latent + text_emb

        golden_embedding = self.unet(encoder_hidden_states.float())

        self.pipeline = self.pipeline.to(torch.float16)
        latent = latent.half()

        original_noise = latent.detach()

        golden_noise = (original_noise.float() + golden_embedding.float()).half()

        original_img = self.pipeline.apply(
            prompt=prompt,
            latents=latent)
        image_numpy = original_img.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        original_img = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
        
        golden_img = self.pipeline.apply(
            prompt=prompt,
            latents=golden_noise)
        image_numpy = golden_img.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        golden_img = DiffusionPipeline.numpy_to_pil(image_numpy)[0]

        
        if config.metric_version == 'PickScore':
            before_rewards, original_scores = reward_model.calc_probs(prompt, original_img)
            golden_rewards, golden_scores = reward_model.calc_probs(prompt, golden_img)
        
        # print(f'prompt:{prompt}')
        # print(f'origin_score:{original_scores}, golden_score:{golden_scores}')

        return original_scores, golden_scores, original_img, golden_img


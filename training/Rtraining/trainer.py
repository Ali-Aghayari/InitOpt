import logging
import math
from typing import Dict, List, Optional, Tuple

import PIL
import PIL.Image
import torch
from diffusers import DiffusionPipeline

from typing import Any, List

from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize)
from transformers import AutoProcessor

from abc import ABC, abstractmethod


def clip_img_transform(size: int = 224):
    return Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

class BaseRewardLoss(ABC):
    """
    Base class for reward functions implementing a differentiable reward function for optimization.
    """

    def __init__(self, name: str, weighting: float):
        self.name = name
        self.weighting = weighting

    @staticmethod
    def freeze_parameters(params: torch.nn.ParameterList):
        for param in params:
            param.requires_grad = False

    @abstractmethod
    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_text_features(self, prompt: str) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        pass

    def process_features(self, features: torch.Tensor) -> torch.Tensor:
        features_normed = features / features.norm(dim=-1, keepdim=True)
        return features_normed

    def __call__(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(prompt)

        image_features_normed = self.process_features(image_features)
        text_features_normed = self.process_features(text_features)

        loss = self.compute_loss(image_features_normed, text_features_normed)
        return loss



class LatentNoiseTrainer:
    """Trainer for optimizing latents with reward losses."""

    def __init__(
        self,
        reward_losses: List[BaseRewardLoss],
        model: DiffusionPipeline,
        n_iters: int,
        n_inference_steps: int,
        seed: int,
        no_optim: bool = False,
        regularize: bool = True,
        regularization_weight: float = 0.01,
        grad_clip: float = 0.1,
        log_metrics: bool = True,
        save_all_images: bool = False,
        imageselect: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        self.reward_losses = reward_losses
        self.model = model
        self.n_iters = n_iters
        self.n_inference_steps = n_inference_steps
        self.seed = seed
        self.no_optim = no_optim
        self.regularize = regularize
        self.regularization_weight = regularization_weight
        self.grad_clip = grad_clip
        self.log_metrics = log_metrics
        self.save_all_images = save_all_images
        self.imageselect = imageselect
        self.device = device
        self.preprocess_fn = clip_img_transform(224)

    def train(
        self,
        latents: torch.Tensor,
        prompt: str,
        optimizer: torch.optim.Optimizer,
        save_dir: Optional[str] = None,
        multi_apply_fn=None,
    ) -> Tuple[PIL.Image.Image, Dict[str, float], Dict[str, float]]:
        best_loss = torch.inf
        best_image = None
        initial_image = None
        initial_rewards = None
        best_rewards = None
        best_latents = None
        latent_dim = math.prod(latents.shape[1:])
        initial_latents = latents.clone()
        for iteration in range(self.n_iters):
            rewards = {}
            optimizer.zero_grad()
            generator = torch.Generator("cuda").manual_seed(self.seed)
            image, SD_latent = self.model.apply(
                latents=latents,
                prompt=prompt,
                generator=generator,
                num_inference_steps=self.n_inference_steps,
                return_latents = True
            )
            if iteration == 0 : 
                init_SD_latent = SD_latent
            if initial_image is None:
                initial_image = image
                image_numpy = (
                    initial_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                )
                initial_image = DiffusionPipeline.numpy_to_pil(image_numpy)[0]

            total_loss = 0
            preprocessed_image = self.preprocess_fn(image)
            for reward_loss in self.reward_losses:
                loss = reward_loss(preprocessed_image, prompt)
                total_loss += loss * reward_loss.weighting
                rewards[reward_loss.name] = loss.item()
            rewards["total"] = total_loss.item()
            total_reward_loss = total_loss.item()
            if self.regularize:
                # compute in fp32 to avoid overflow
                latent_norm = torch.linalg.vector_norm(latents).to(torch.float32)
                log_norm = torch.log(latent_norm)
                regularization = self.regularization_weight * (
                    0.5 * latent_norm**2 - (latent_dim - 1) * log_norm
                )
                rewards["norm"] = latent_norm.item()
                total_loss += regularization.to(total_loss.dtype)

            if total_reward_loss < best_loss:
                best_loss = total_reward_loss
                best_image = image
                best_rewards = rewards
                best_latents = latents.detach().cpu()
                best_SD_latent = SD_latent

            if iteration != self.n_iters - 1 and not self.imageselect:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(latents, self.grad_clip)
                optimizer.step()

            if initial_rewards is None:
                initial_rewards = rewards

        image_numpy = best_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        best_image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]

        return initial_image, best_image_pil, initial_latents, best_latents, init_SD_latent, best_SD_latent

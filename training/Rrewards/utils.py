from typing import Any, List

import torch
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize)
from transformers import AutoProcessor

from .aesthetic import AestheticLoss
from .base_reward import BaseRewardLoss
from .clip import CLIPLoss
from .hps import HPSLoss
from .imagereward import ImageRewardLoss
from .pickscore import PickScoreLoss


def get_reward_losses(dtype: torch.dtype, device: torch.device, cache_dir: str
) -> List[BaseRewardLoss]:
    tokenizer = AutoProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
    )
    reward_losses = []
    reward_losses.append(
        ImageRewardLoss(
            1,
            dtype,
            device,
            cache_dir,
            memsave=False,
        )
    )
    reward_losses.append(
        CLIPLoss(
            0.01,
            dtype,
            device,
            cache_dir,
            tokenizer,
            memsave=False,
        )
    )
    reward_losses.append(
        PickScoreLoss(
            0.05,
            dtype,
            device,
            cache_dir,
            tokenizer,
            memsave=False,
        )
    )
    reward_losses.append(
        HPSLoss(5, dtype, device, cache_dir, memsave=False)
    )
    return reward_losses


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

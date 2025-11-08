import json
import numpy as np
import math
import csv
import random
import argparse
import torch
import os
import torch.distributed as dist

from torchvision import transforms
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, \
    DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, \
    DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler

from torch.nn.parallel import DistributedDataParallel as DDP

from reward_model.eval_pickscore import PickScore
import hpsv2
import ImageReward as RM
from reward_model.aesthetic_scorer import AestheticScorer
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from pytorch_lightning import seed_everything


from ReNO import apply_reno
device = torch.device('cuda')

from Rmodels import get_model
from Rrewards import get_reward_losses




def get_args():
    # pick: test_unique_caption_zh.csv       draw: drawbench.csv
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt_dataset", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    args =  parser.parse_args()
    return args


def load_prompt(path, seed_path, prompt_version):
    if prompt_version == 'pick':
        prompts = []
        with open(path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])

        prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list
    
    elif prompt_version == 'draw':
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "Prompts":
                    continue
                prompts.append(row[0])

        prompts = prompts[0:200]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)

        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)
        return prompts, seed_list
    
    else:
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])
        # prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list


def load_pick_prompt(path):
    prompts = []
    seeds = []
    with open(path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        
        for row in content:
            data = eval(row)
            prompts.append(data['caption'])
            seeds.append(data['seed'])
        
    return prompts, seeds


# PICK_MODEL = PickScore()
HPSV2_MODEL = hpsv2
# IR_MODEL = RM.load("ImageReward-v1.0")
# AES_MODEL = AestheticScorer(dtype = torch.float32)



def cal_score(prompt, image):

    # _, pick_score = PICK_MODEL.calc_probs(prompt, image)

    hpsv2_score = HPSV2_MODEL.score([image], prompt, hps_version="v2.1")[0]

    # ir_score = IR_MODEL.score(prompt, image)

    # aes_score = AES_MODEL(image)

    # return [pick_score.item(), float(hpsv2_score), ir_score, aes_score.item()]
    return [float(hpsv2_score)]


import tempfile
import shutil
def safe_savez(path, *arrays):
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False) as tmp:
        np.savez_compressed(tmp, *arrays)   # compressed, smaller file
        tmp_path = tmp.name
    shutil.move(tmp_path, path)  # atomic replace (safe rename)

if __name__ == '__main__':

    seed_everything(0)

    dtype = torch.float16
    args = get_args()

    prompt_list, seed_list = load_pick_prompt(
        path='data/train_custom.json'
    )

    model = "sd-turbo"
    cache_dir = "./cache"
    
    before_score, after_score, positive = 0, 0, 0

    device = torch.device("cuda")
    dtype = torch.float16

    reward_losses = get_reward_losses(dtype, device, cache_dir)
    # Get model and noise trainer
    pipe = get_model(
        model, dtype, device, cache_dir, False, False
    )


    with open('renoSDlatent.json', 'w+') as file:
        for idx, prompt in enumerate(prompt_list):
            random_seed = seed_list[idx]  # 拿到seed_list中的 seed

            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # ---- skip if file already exists ----
            if random_seed is not None:
                filename = os.path.join(output_dir, f"{idx}_{random_seed}.npz")
            else:
                filename = os.path.join(output_dir, f"{idx}.npz")

            if os.path.exists(filename):
                continue  # 🔥 skip this iteration


            original_img, optim_img, _, _, original_latents, inversion_latents = apply_reno(prompt, random_seed, f"{args.output_dir}/reno/", reward_losses, pipe)
            # original_scores = cal_score(prompt, original_img)
            # optimized_scores = cal_score(prompt, optim_img)

            data_pair = (original_latents, inversion_latents, idx)

            output_dir = args.output_dir

            if not os.path.exists(output_dir): # noise_pairs_dreamshaper_step_4_10_100_pick
                os.mkdir(output_dir)

            filename = f"{idx}_{random_seed}.npz" if random_seed is not None else f"{idx}.npz"
            path = os.path.join(output_dir, filename)
            safe_savez(path, *data_pair)


            # pick_score --> hpsv2 --> ir --> aes
            data = {
                'index': idx,
                'seed': random_seed,
                'caption': prompt,
                'original_score_list': 0,
                'optimized_score_list': 1
                
            }
            json.dump(data, file)
            file.write('\n')


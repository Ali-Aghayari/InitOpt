import torch
import clip
import requests
import os
import csv
import argparse
import numpy as np
from PIL import Image

import hpsv2
import ImageReward as RM
from reward_model.aesthetic_scorer import AestheticScorer
from tqdm.auto import tqdm
from reward_model.eval_pickscore import PickScore
from transformers import CLIPImageProcessor, AutoTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer_example(images, prompt, condition, clip_model, clip_processor, tokenizer, device):
    def _process_image(image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open( image )
        image = image.convert("RGB")
        pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(caption):
        input_ids = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids
    
    image_inputs = torch.concatenate([_process_image(images[0]).to(device), _process_image(images[1]).to(device)])
    text_inputs = _tokenize(prompt).to(device)
    condition_inputs = _tokenize(condition).to(device)

    with torch.no_grad():
        text_features, image_0_features, image_1_features = clip_model(text_inputs, image_inputs, condition_inputs)
        image_0_features = image_0_features / image_0_features.norm(dim=-1, keepdim=True)
        image_1_features = image_1_features / image_1_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = clip_model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_1_features))
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)[0]

    return probs.cpu().tolist()


def load_single_image_list(file_path):
    image_list = os.listdir(file_path)
    res = []
    for idx in range(len(image_list)):
        res.append(Image.open(os.path.join(file_path, f'{idx}.png')))
    return res


def load_prompt(path, dataset_version):
    if dataset_version == 'pick':
        prompts = []
        with open(path, 'r') as file:
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
    elif dataset_version == 'draw':
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
    elif dataset_version == 'hpd':
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

    elif dataset_version == "hpsv2-anime" or dataset_version == "hpsv2-photo" or dataset_version == "hpsv2-concept" or dataset_version == "hpsv2-painting":
        prompts = []
        seed_list = []
        with open(path, 'r') as file:
            contents = file.readlines()
            for line in contents:
                parts = [part.strip() for part in line.split(',')]
                prompt = ', '.join(parts[1:-1])
                seed = int(parts[-1])
                prompts.append(prompt)
                seed_list.append(int(seed))

        return prompts

    elif dataset_version == 'geneval':
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])

        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
    else:
        return None

    return prompts


def load_image(path):
    origin_list = os.listdir(os.path.join(path, 'origin'))
    optim_list = os.listdir(os.path.join(path, 'optim'))


    origin_image_list, optim_image_list = [], []

    for idx in range(len(origin_list)):
        origin_image_list.append(Image.open(os.path.join(path,'origin',f'{idx}.png')))

    for idx in range(len(optim_list)):

        optim_image_list.append(Image.open(os.path.join(path,'optim',f'{idx}.png')))
    return origin_image_list, optim_image_list

def cal_score(prompt_list, image_list, metric_version):
    prompt_list = prompt_list[:len(image_list)]
    assert len(prompt_list) == len(image_list)
    total_score = 0
    score_list = []
    if metric_version == 'PickScore':
        reward_model = PickScore()
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            _, score = reward_model.calc_probs(prompt, image)
            total_score += score
            score_list.append(score)

    elif metric_version == 'HPSv2':
        os.environ['HPS_ROOT'] = "./models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b/"
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = hpsv2.score([image], prompt, hps_version="v2.1")
            print(score)
            total_score += score[0]
            score_list.append(score)

    elif metric_version == 'ImageReward':
        reward_model = RM.load("ImageReward-v1.0")
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = reward_model.score(prompt, image)
            # print(score)
            total_score += score
            score_list.append(score)

    elif metric_version == 'AES':
        reward_model = AestheticScorer(dtype = torch.float32)
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = reward_model(image)
            total_score += score[0]
            score_list.append(score[0])
    
    elif metric_version == 'CLIP':
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            image = preprocess(image).unsqueeze(0).to(DEVICE)
            text = clip.tokenize(prompt, context_length=77, truncate=True).to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(image).cpu().numpy()
                text_features = model.encode_text(text).cpu().numpy()

                image_features = image_features / np.sqrt(np.sum(image_features**2, axis=1, keepdims=True))
                text_features = text_features / np.sqrt(np.sum(text_features**2, axis=1, keepdims=True))

                score = np.mean(2.5 * np.clip(np.sum(image_features * text_features, axis=1), 0, None))

            total_score += score
            score_list.append(score)

    else:
        raise NotImplementedError

    return total_score/len(prompt_list), score_list


def cal_mps_score(prompt_list, original_image_list, optim_image_list):
    prompt_list = prompt_list[:len(original_image_list)]

    assert len(prompt_list) == len(original_image_list)
    
    original_total_score = 0
    optim_total_score = 0

    original_score_list = []
    optim_score_list = []

    condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things." 

    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    image_processor = CLIPImageProcessor.from_pretrained(processor_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(processor_name_or_path, trust_remote_code=True)
    model_ckpt_path = "./metric/mps/outputs/MPS_overall_checkpoint.pth"
    model = torch.load(model_ckpt_path)
    model.model.text_model.eos_token_id=2
    model.eval().to(DEVICE)

    for prompt, original_image, optim_image in tqdm(zip(prompt_list, original_image_list, optim_image_list)):
        score_list = infer_example([original_image, optim_image], prompt, condition, model, image_processor, tokenizer, DEVICE)
        
        original_total_score += score_list[0]
        optim_total_score += score_list[1]

        original_score_list.append(score_list[0])
        optim_score_list.append(score_list[1])

    return original_total_score/len(prompt_list), optim_total_score/len(prompt_list), \
            original_score_list, optim_score_list

def get_args():
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--benchmark', default='pick', choices=['pick', 'draw', 'hpd', 'geneval', 'hpsv2-anime', 'hpsv2-photo', 'hpsv2-concept', 'hpsv2-painting'], type=str)
    
    # MPS can be used in mps file, not here
    parser.add_argument("--metric-version", default='PickScore', choices=['PickScore', 'HPSv2', 'AES', 'ImageReward', 'CLIP', 'MPS'],
                        type=str)
   
    args = parser.parse_args()

    print("generating config:")
    print(f"Config: {args}")
    print('-' * 100)

    return args


if __name__ == '__main__':

    args = get_args()

    
    dataset_version = args.benchmark  #['pick', 'draw', 'hpsv2-anime', 'hpsv2-photo', 'hpsv2-concept', 'hpsv2-painting', 'hpd', 'geneval']
    metric_version = args.metric_version     #['PickScore', 'HPSv2', 'AES', 'ImageReward',]

    if dataset_version == 'pick':
        prompt_path = "pickscore.csv"
    elif dataset_version == 'draw':
        prompt_path = "drawbench.csv"
    elif dataset_version == 'hpd': 
        prompt_path = "HPD_prompt.csv"
    elif dataset_version == 'geneval':
        prompt_path = "Geneval_prompt.csv"
    elif dataset_version == 'hpsv2-anime':
        prompt_path = "anime.txt"
    elif dataset_version == 'hpsv2-photo':
        prompt_path = "photo.txt"
    elif dataset_version == 'hpsv2-concept':
        prompt_path = "concept-art.txt"
    elif dataset_version == 'hpsv2-painting':
        prompt_path = "paintings.txt"
    else:
        raise NotImplementedError
    
    prompt_list = load_prompt(prompt_path, dataset_version)

    #load image_list
    # image_dir_path = './exp_data/draw_sdxl_main'
    image_dir_path = "xxx" # TODO: change to the path of the image

    print(f'load image from path: {image_dir_path}')


    image_origin, image_optim = load_image(image_dir_path)
    # image_optim = image_origin
    print('loaded image ...')


    if args.metric_version != 'MPS':
        # origin metric
        print(f'start cal {metric_version} about origin images')
        origin_score, origin_score_list = cal_score(prompt_list, image_origin, metric_version)

        # #optim metric
        print(f'start cal {metric_version} about optim images')
        optim_score, optim_score_list = cal_score(prompt_list, image_optim, metric_version)
    else:
        print(f'start cal {metric_version} about original and optimized images')
        origin_score, optim_score, origin_score_list, optim_score_list = cal_mps_score(prompt_list, image_origin, image_optim)

    positive = 0
    for origin,optim in zip(origin_score_list, optim_score_list):
        if origin < optim:
            positive += 1
    print(f'wining rate between origin and optim: {positive/len(optim_score_list)}')
    print(f'origin_score:{origin_score}      optim_score:{optim_score}')


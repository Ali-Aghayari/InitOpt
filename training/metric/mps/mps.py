import os
import csv
import torch
import argparse

from tqdm import tqdm
from transformers import CLIPImageProcessor, AutoTokenizer
from PIL import Image

device = "cuda"


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
    if dataset_version == 'pick_score':
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
    elif dataset_version == 'drawBench':
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
        prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
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

def load_our_image(origin_path, golden_path):
    origin_image_list, golden_image_list = [], []

    original_list = os.listdir(origin_path)
    golden_list = os.listdir(golden_path)


    for image in original_list:
        origin_image_list.append(Image.open(os.path.join(origin_path, image)))
    for image in golden_list:
        golden_image_list.append(Image.open(os.path.join(golden_path, image)))
    
    return origin_image_list, golden_image_list


def load_image(path):
    origin_list = os.listdir(os.path.join(path, 'origin'))
    optim_list = os.listdir(os.path.join(path, 'optim'))
    
    origin_image_list, optim_image_list = [], []
    for idx in range(len(origin_list)):
        # set path_origin to baseline reslut path
        # origin_image_list.append(Image.open(os.path.join(path_origin,'origin',f'{idx}.png')))
        # set path_origin to path
        origin_image_list.append(Image.open(os.path.join(path,'origin',f'{idx}.png')))
    for idx in range(len(optim_list)):
        optim_image_list.append(Image.open(os.path.join(path,'optim',f'{idx}.png')))
    return origin_image_list, optim_image_list


def get_args():
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--benchmark', default='pick_score', choices=['pick_score', 'drawBench', 'hpd', 'geneval'], type=str)
    parser.add_argument('--model', default=None, type=str)

    args = parser.parse_args()

    print("generating config:")
    print(f"Config: {args}")
    print('-' * 100)

    return args


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
    model_ckpt_path = "/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/metric/mps/outputs/MPS_overall_checkpoint.pth"
    model = torch.load(model_ckpt_path)
    model.model.text_model.eos_token_id=2
    model.eval().to(device)

    for prompt, original_image, optim_image in tqdm(zip(prompt_list, original_image_list, optim_image_list)):
        score_list = infer_example([original_image, optim_image], prompt, condition, model, image_processor, tokenizer, device)
        
        original_total_score += score_list[0]
        optim_total_score += score_list[1]

        original_score_list.append(score_list[0])
        optim_score_list.append(score_list[1])

    return original_total_score/len(prompt_list), optim_total_score/len(prompt_list), \
            original_score_list, optim_score_list

if __name__ == '__main__':

    args = get_args()

    #load prompt_list
    dataset_version = args.benchmark  #['pick_score', 'drawBench', 'hpd']

    if dataset_version == 'pick_score':
        prompt_path = "/data/zhouzikai/project_data/NoiseModel/datasets/prompt/test_unique_caption_zh.csv"
    elif dataset_version == 'drawBench':
        prompt_path = "/data/zhouzikai/project_data/NoiseModel/datasets/prompt/drawbench.csv"
    elif dataset_version == 'hpd': 
        prompt_path = "/data/zhouzikai/project_data/NoiseModel/datasets/prompt/HPD_prompt.csv"
    elif dataset_version == 'geneval':
        prompt_path = "/data/zhouzikai/project_data/NoiseModel/datasets/prompt/Geneval_prompt.csv"
    else:
        raise NotImplementedError
    
    prompt_list = load_prompt(prompt_path, dataset_version)

    #load image_list
    # image_dir_path = './exp_data/draw_sdxl_main'
    image_dir_path = '/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/Geneval/SDXL/output_SDXL_step_10_geneval'
    print(f'load image from path: {image_dir_path}')


    if args.model == 'ours':
        origin_path = "/data/zhouzikai/project_data/NoiseModel/datasets/drawbench/output_SDXL_50_draw/origin/"
        golden_path = "/data/zhouzikai/project_data/NoiseModel/datasets/drawbench/SDXL/APG-50/optim"
        image_origin, image_optim = load_our_image(origin_path, golden_path)
    else:
        image_origin, image_optim = load_image(image_dir_path)

    # image_optim = image_origin
    print('loaded image ...')

    print(f'start cal MPS about original and optimized images')
    origin_score, optim_score, origin_score_list, optim_score_list = cal_mps_score(prompt_list, image_origin[:101], image_optim[:101])

    positive = 0
    for origin,optim in zip(origin_score_list, optim_score_list):
        if origin < optim:
            positive += 1
    print(f'wining rate between origin and optim: {positive/len(optim_score_list)}')
    print(f'origin_score:{origin_score}      optim_score:{optim_score}')
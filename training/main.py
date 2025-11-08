import json
import os
import random
import argparse
import numpy as np
import accelerate
import torch
import torch.distributed as dist

from diffusers import DDIMScheduler

from reward_model.eval_pickscore import PickScore
from torch.utils.data import DataLoader

from solver import solver_dict
from noise_dataset import NoiseDataset
from reward_model.eval_pickscore import PickScore
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from diffusers import DiffusionPipeline

from tqdm import tqdm

from Rmodels import get_model


DEVICE = torch.device("cuda" if torch.cuda else "cpu")

def get_args():
    parser = argparse.ArgumentParser()

    # model and dataset construction
    parser.add_argument('--pipeline', default='SDT', choices=['SDT'], type=str)

    parser.add_argument('--type', default='1', choices=['1', '2', '3'], type=str)
    
    parser.add_argument("--benchmark-type", default='pick', choices=['pick', 'draw'], type=str)
    parser.add_argument("--train", default=False, type=bool)
    parser.add_argument("--evaluate", default=False, type=bool)    

    # hyperparameters
    parser.add_argument('--pick', default=False, type=bool)

    parser.add_argument("--all-file", default=False, type=bool)
    parser.add_argument("--epochs", default=10, type=int)  # more iterations, less epochs ==> 3
    parser.add_argument("--batch-size", default=32, type=int) # verify ESVD, SVD, EUnet with prompt[0]  
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--metric-version", default='PickScore', choices=['PickScore', 'HPSv2', 'AES', 'ImageReward'], type=str)

    # path configuration
    parser.add_argument("--prompt-path", default='./trainColor.json', type=str)
    parser.add_argument("--data-dir", default="./data/train_npz/", type=str)
    parser.add_argument('--pretrained-path', type=str, default='./training/checkpoints')
    parser.add_argument('--save-ckpt-path', type=str, default='./training/checkpoints/sdt')

    # discard the bad samples
    parser.add_argument("--discard", default=False, type=bool)

    args = parser.parse_args()

    print("generating config:")
    print(f"Config: {args}")
    print('-' * 100)

    return args


if __name__ == '__main__':
    dtype = torch.float16
    args = get_args()

    # Ensure required directories exist (pretrained and checkpoint)
    # Note: If a file path is passed instead, please pass its directory path.
    if isinstance(args.pretrained_path, str):
        os.makedirs(args.pretrained_path, exist_ok=True)
    if isinstance(args.save_ckpt_path, str):
        dir_name = args.save_ckpt_path.split('/')[-2]
        os.makedirs(f'./training/checkpoints/{dir_name}', exist_ok=True)

    # construct the diffusion models and human perference models
    reward_model = PickScore()

    # stabilityai/stable-diffusion-2-1  Lykon/dreamshaper-xl-v2-turbo stabilityai/sdxl-turbo
    if args.pipeline == 'SDT':
        pipeline = get_model("sd-turbo", dtype, DEVICE, "./cache", False, False)

    else:
        print(f'Pipeline {args.pipeline} doesn`t exist!')
        assert False


    solver = solver_dict[f"ReNO{args.type}"](
            pipeline=pipeline,
            config=args
        )



    # construct the dataset
    if args.train:
        NoiseDataset_100 = NoiseDataset(
            discard=args.discard,
            pick=args.pick,
            all_file=args.all_file,
            data_dir=args.data_dir,
            prompt_path=args.prompt_path)

        from sklearn.model_selection import StratifiedShuffleSplit

        labels = [0 for i in range(len(NoiseDataset_100))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]

        trainset = torch.utils.data.Subset(NoiseDataset_100, train_indices)
        valset = torch.utils.data.Subset(NoiseDataset_100, valid_indices)

        print(f'training set size: {len(trainset)}')
        print(f'validation set size: {len(valset)}')

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)


        solver.train(
            train_loader,
            val_loader,
            total_epoch=args.epochs,
            save_path=args.save_ckpt_path)

    if args.evaluate:
        random_seed = 120
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        
        all_original, all_golden = [], []
        all_optim, all_optimN = [], []

        eval_dataset = NoiseDataset(
            discard=args.discard,
            pick=args.pick,
            all_file=args.all_file,
            data_dir=args.data_dir,
            prompt_path=args.prompt_path
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,          # keep small to avoid CUDA OOM
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        

        # with open("color_train.txt") as f:
        #     for step, prompt in enumerate(f):

        with torch.no_grad():
            pbar = tqdm(eval_loader, desc="Evaluating", dynamic_ncols=True)
            for step, (original_noise, optimized_noise, prompt) in enumerate(pbar, 1):
                prompt = prompt[0].strip()

                random_latent = torch.randn(1, 4, 512//8, 512//8, dtype=dtype).to(DEVICE)
                random_latent = original_noise.to(DEVICE)
                optim_latent = optimized_noise.to(DEVICE)

                # your solver call
                original_scores, golden_scores, original_img, golden_img = solver.generate(
                    random_latent,
                    None,
                    reward_model,
                    config=args,
                    prompt=prompt,
                )

                optim_scores, optimN_scores, optim_img, optimN_img = solver.generate(
                    optim_latent,
                    None,
                    reward_model,
                    config=args,
                    prompt=prompt,
                )

                base_folder = "./images"
                subfolder_name = f"{step:07d}_" + "".join(c for c in prompt[:150] if c.isalnum() or c in (" ", "_", "-")).rstrip()
                subfolder_path = os.path.join(base_folder, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)
                
                golden_img_path = os.path.join(subfolder_path, f'golden.png')
                original_img_path = os.path.join(subfolder_path, f'original.png')
                optim_img_path = os.path.join(subfolder_path, f'optim.png')
                optimN_img_path = os.path.join(subfolder_path, f'optimN.png')

                golden_img.save(golden_img_path)
                original_img.save(original_img_path)
                optim_img.save(optim_img_path)
                optimN_img.save(optimN_img_path)

                all_original.append(original_scores.item())
                all_golden.append(golden_scores.item())
                all_optim.append(optim_scores.item())
                all_optimN.append(optimN_scores.item())

        print("mean original:", np.mean(all_original))
        print("mean golden:", np.mean(all_golden))
        print("mean optim:", np.mean(all_optim))
        print("mean optim noised:", np.mean(all_optimN))

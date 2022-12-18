import os
import cv2
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import argparse
from optimizedSD.optimized_img2img import optimized_image_two_image
from optimizedSD.sequence_generation import rename_sequence
import random
import shutil

# python optimizedSD/sequence_chain.py --prompt "two witches making potion" --strength 0.1 --W 512 --H 512 --turbo

# TODO: rename file from seed_0 during main loop in case stopping early so dont have to rename manually
# TODO: include source image at beginning of directory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, default="assets/mom_and_nancy.jpg")
    parser.add_argument("--strength", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    opt = parser.parse_args()

    img_name_encoded = opt.image_file.split("/")[-1].split(".")[0].replace(" ", "_").lower()
    prompt_encoded = opt.prompt.replace(" ", "_").lower()
    outdir = f"outputs/sequence-chain/{img_name_encoded}/{prompt_encoded}_{opt.strength}_{opt.seed}_{opt.W}_{opt.H}"
    final_outdir = f"outputs/sequence-chain-processed/{img_name_encoded}/{prompt_encoded}_{opt.strength}_{opt.seed}_{opt.W}_{opt.H}"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    seed = opt.seed
    source_image = opt.image_file
    for i in range(opt.n):
        source_image = optimized_image_two_image(    
            outdir=outdir,
            seed=seed+i, init_img=source_image, H=opt.H, W=opt.W, device='cuda',
            unet_bs=1, turbo=opt.turbo, precision="autocast", n_samples=1,
            n_rows=0, from_file=None, prompt=opt.prompt, strength=opt.strength,
            ddim_steps=50, n_iter=1, scale=10.0, ddim_eta=0.0,
            sampler="ddim", format="png"
        )[0]
    rename_sequence(outdir + "/" + prompt_encoded, final_outdir)


if __name__ == '__main__':
    main()
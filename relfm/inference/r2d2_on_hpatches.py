"""Inference module of R2D2-like models on HPatches dataset."""
"""Evaluation script for the HPatches dataset."""

import os
from os.path import join, exists, expanduser
from genericpath import isdir
from glob import glob
import numpy as np
from PIL import Image
import torch

from relfm.utils.paths import REPO_PATH
from lib.r2d2.extract import extract_keypoints_modified, load_network
from relfm.utils.log import print_update, tqdm_iterator
from relfm.utils.visualize import show_images_with_keypoints
from relfm.utils.matching import evaluate_matching_with_rotation, analyze_result


def configure_save_dir(output_base_dir, ckpt_path, dataset_name="hpatches"):
    """Configures the save directory for the evaluation results."""

    ckpt_name = os.path.basename(ckpt_path).split(".pt")[0]
    save_dir = join(output_base_dir, dataset_name, ckpt_name)
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluation script for the HPatches dataset.",
    )
    parser.add_argument(
        "--data_dir", "-d", type=str,
        default=join(REPO_PATH, "data/hpatches-sequences-release/"),
        help="Path to the HPatches dataset.",
    )
    parser.add_argument(
        "--model_ckpt_path", "-m", type=str,
        default=join(REPO_PATH, "checkpoints/r2d2_WASF_N16.pt"),
        help="Path to the (pretrained) R2D2 model checkpoint.",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str,
        default=join(expanduser("~"), "outputs/rotation-equivariant-lfm"),
        help="Base output directory in which to store the results.",
    )
    parser.add_argument(
        "--gap_between_rotations", "-g", type=int,
        default=15,
        help="Number of degrees between two consecutive rotations.",
    )
    parser.add_argument(
        "--debug", "-D", action="store_true",
        help="Whether to run only on 1 sample sequence to debug.",
    )
    parser.add_argument(
        "--num_keypoints", "-k", type=int,
        default=1000,
        help="Number of keypoints to extract from each image.",
    )
    parser.add_argument(
        "--imsize", "-i", type=int,
        default=300,
        help="Size of the images to be used for the evaluation.",
    )
    parser.add_argument(
        "--seq_prefix", "-s", type=str,
        default=None,
        help="Prefix of the sequence to be used for the evaluation.",
    )
    args = parser.parse_args()

    assert isdir(args.data_dir), \
        f"Could not find the HPatches dataset at {args.data_dir}."
    
    assert exists(args.model_ckpt_path), \
        f"Could not find the R2D2 model checkpoint at {args.model_ckpt_path}."
    
    # create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # configure save directory
    save_dir = configure_save_dir(args.output_dir, args.model_ckpt_path, dataset_name="hpatches")

    # load network
    print_update("Loading network.")
    model = load_network(args.model_ckpt_path)

    sequences = sorted(glob(join(args.data_dir, "*")))
    if args.seq_prefix is not None:
        sequences = [s for s in sequences if args.seq_prefix in s]
    rotations = np.arange(0, 360 + 1, args.gap_between_rotations, dtype=int)
    print_update(f"Generating predictions on {len(sequences)} sequences for {len(rotations)} rotations per image.")
    counter = 1
    for sequence in sequences:
        # set path to the source image
        img1_path = join(sequence, "1.ppm")
        img1 = Image.open(img1_path)
        original_width, original_height = img1.size
        # img1 = img1.resize((args.imsize, args.imsize))

        # generate outputs for source image
        outputs = extract_keypoints_modified([img1], model, top_k=args.num_keypoints, verbose=False)[0]
        sequence_name = os.path.basename(sequence)
        os.makedirs(join(save_dir, sequence_name), exist_ok=True)
        # save_path = join(save_dir, sequence_name, "1.pt")
        # torch.save(outputs, save_path)
        save_path = join(save_dir, sequence_name, "1.npy")
        np.save(save_path, outputs)

        # possible indices of the target images
        img2_indices = np.arange(2, 7)
        # load all target images at once
        # img2s = [Image.open(join(sequence, f"{i}.ppm")).resize((args.imsize, args.imsize)) for i in img2_indices]
        img2s = [Image.open(join(sequence, f"{i}.ppm")) for i in img2_indices]

        # load all homographies
        Hs = [np.loadtxt(join(sequence, f"H_1_{i}")) for i in img2_indices]

        rotation_grid, img2_indices_grid  = np.meshgrid(rotations, img2_indices)
        rotation_grid, img2_indices_grid = rotation_grid.flatten(), img2_indices_grid.flatten()

        iterator = tqdm_iterator(range(len(rotation_grid)), desc=f"Generating predictions for {sequence_name}")
        for i in iterator:
            rotation, img2_index = rotation_grid[i], img2_indices_grid[i]
            img2 = img2s[img2_index - 2]
            img2_rotated = img2.rotate(rotation)

            H = Hs[img2_index - 2].copy()

            # # apply resing effect on H
            # sx = args.imsize / original_width
            # sy = args.imsize / original_height

            # H_for_scaling = np.array([
            #     [sx, 0., 0.],
            #     [0., sy, 0.],
            #     [0., 0., 1.],
            # ])
            # H = H_for_scaling @ Hs[img2_index - 2]

            outputs = extract_keypoints_modified([img2_rotated], model, top_k=args.num_keypoints, verbose=False)[0]
            outputs.update(
                {
                    "rotation": rotation,
                    "H": H,
                }
            )
            # save_path = join(save_dir, sequence_name, f"{img2_index}_rotation_{rotation}.pt")
            # torch.save(outputs, save_path)
            save_path = join(save_dir, sequence_name, f"{img2_index}_rotation_{rotation}.npy")
            np.save(save_path, outputs)
        
        print(f"Finished processing sequence {sequence_name} ({counter}/{len(sequences)}).")
        counter += 1 

        if args.debug:
            break

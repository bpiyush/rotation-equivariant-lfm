"""Inference module of R2D2-like models on HPatches dataset."""

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
from relfm.utils.visualize import show_images_with_keypoints, check_kps_with_homography
from relfm.utils.matching import evaluate_matching_with_rotation, analyze_result
from relfm.utils.geometry import append_rotation_to_homography, apply_homography_to_keypoints, resize, apply_clean_rotation
from lib.r2d2.nets.patchnet import Quad_L2Net_ConfCFS
from lib.r2d2.nets.patchnet_equivariant import (
    Discrete_Quad_L2Net_ConfCFS,
    Steerable_Quad_L2Net_ConfCFS,
)


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
        default=join(REPO_PATH, "trained_models/epoch_3_SO2_4x16_1x32_1x64_2x128.pt"),
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
        "--downsize", action="store_true",
        help="Whether to downsize the images before extracting keypoints.",
    )
    parser.add_argument(
        "--seq_prefix", "-s", type=str,
        default=None,
        help="Prefix of the sequence to be used for the evaluation.",
    )
    parser.add_argument(
        "--sanity_check", action="store_true",
        help="Whether to run sanity checks on the extracted keypoints.",
    )
    parser.add_argument(
        "--crop_post_rotation", action="store_true",
        help="Whether to crop the keypoints after applying the rotation.",
    )
    args = parser.parse_args()

    assert isdir(args.data_dir), \
        f"Could not find the HPatches dataset at {args.data_dir}."
    
    assert exists(args.model_ckpt_path), \
        f"Could not find the R2D2 model checkpoint at {args.model_ckpt_path}."
    
    # create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # configure save directory
    save_dir = configure_save_dir(
        args.output_dir, args.model_ckpt_path, dataset_name="hpatches",
    )

    # load network
    print_update("Loading network.")
    if "r2d2" in args.model_ckpt_path:
        print("Loading R2D2 model.")
        model = load_network(args.model_ckpt_path)
        model = model.eval()
    else:
        print("Loading R2D2-like equivariant model.")
        checkpoint = torch.load(args.model_ckpt_path, map_location="cpu")
        model = eval(checkpoint['net'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.eval()

    # load sequence paths
    sequences = sorted(glob(join(args.data_dir, "*")))
    if args.seq_prefix is not None:
        sequences = [s for s in sequences if args.seq_prefix in s]
    rotations = np.arange(0, 360 + 1, args.gap_between_rotations, dtype=int)
    print_update(
        f"Generating predictions on {len(sequences)} "\
            f"sequences for {len(rotations)} rotations per image.",
    )

    counter = 1
    for sequence in sequences:
        
        # store sequence name
        sequence_name = os.path.basename(sequence)

        # load source image
        img1_path = join(sequence, "1.ppm")
        img1_raw = Image.open(img1_path)

        # define base homography for source image
        H1_raw = np.eye(3)

        # possible indices of the target images
        img2_indices = np.arange(2, 7)

        # load all target images at once
        img2s_raw = [Image.open(join(sequence, f"{i}.ppm")) for i in img2_indices]

        # define base homography for each target image
        H2s_raw = [np.eye(3) for _ in img2s_raw]

        if args.downsize:
            # downsize the source image to (args.imsize, args.imsize)
            img1_resized, H1_raw = resize(img1_raw, args.imsize, args.imsize)

            # downsize the target images to (args.imsize, args.imsize)
            img2s_resized = []
            for j in range(len(img2s_raw)):
                img, H2s_raw[j] = resize(img2s_raw[j], args.imsize, args.imsize)
                img2s_resized.append(img)
        else:
            img1_resized = img1_raw
            img2s_resized = img2s_raw

        # load all homographies
        H1to2s = [np.loadtxt(join(sequence, f"H_1_{i}")) for i in img2_indices]

        rotation_grid, img2_indices_grid  = np.meshgrid(rotations, img2_indices)
        rotation_grid, img2_indices_grid = rotation_grid.flatten(), img2_indices_grid.flatten()

        iterator = tqdm_iterator(
            range(len(rotation_grid)), desc=f"Generating predictions for {sequence_name} ({counter}/{len(sequences)})\t\t",
        )
        for i in iterator:
            rotation, img2_index = rotation_grid[i], img2_indices_grid[i]

            img1 = img1_resized.copy()

            # index target image
            img2 = img2s_resized[img2_index - 2].copy()

            # load base homography for source and target image
            H1 = H1_raw.copy()
            H2 = H2s_raw[img2_index - 2].copy()

            if args.crop_post_rotation:
                # center crop the source image according to the rotation
                # NOTE: this does not rotate the image, only crops based on rotation
                _, _, img1_transformed, H1 = apply_clean_rotation(
                    image=img1, degrees=rotation, H=H1,
                )

                # rotate + center crop the target image
                # NOTE: this applies rotation and then cropping
                img2_transformed, H2, _, _ = apply_clean_rotation(
                    image=img2, degrees=rotation, H=H2,
                )
            else:
                img1_transformed = img1
                img2_transformed = img2.rotate(rotation)
                H2 = append_rotation_to_homography(H2, rotation, img1.size[0], img1.size[1])

            # transform the homography accordingly
            H = H1to2s[img2_index - 2].copy()
            H_transformed = H2 @ H @ np.linalg.inv(H1)

            if args.sanity_check and rotation in [0, 30, 45, 90, 135, 180]:
                check_kps_with_homography(
                    img1=img1_transformed,
                    img2=img2_transformed,
                    H=H_transformed,
                    save=True,
                    save_path=f"./sample_images/sanity_check_kps_rotation_{rotation}.png",
                )

            # >>> generate outputs for source image and save them
            outputs = extract_keypoints_modified(
                [img1_transformed], model, top_k=args.num_keypoints, verbose=False,
            )[0]
            outputs.update(
                {
                    "rotation": rotation,
                    "H": H_transformed,
                    "H1": H1,
                }
            )
            kps1 = outputs["keypoints"]
            # save outputs
            save_path = join(
                save_dir, sequence_name, f"1_rotation_{rotation}.npy",
            )
            np.save(save_path, outputs)

            # >>> generate outputs for target image, rotated by `rotation` degrees
            outputs = extract_keypoints_modified(
                [img2_transformed], model, top_k=args.num_keypoints, verbose=False,
            )[0]
            outputs.update(
                {
                    "rotation": rotation,
                    "H": H_transformed,
                    "H2": H2,
                }
            )
            kps2 = outputs["keypoints"]
            # save outputs
            save_path = join(
                save_dir, sequence_name, f"{img2_index}_rotation_{rotation}.npy",
            )
            np.save(save_path, outputs)

            if args.sanity_check and rotation in [45]:
                check_kps_with_homography(
                    img1=img1_transformed,
                    img2=img2_transformed,
                    H=H_transformed,
                    kps=kps1,
                    save=True,
                    save_path=f"./sample_images/sanity_check_computed_kps_rotation_{rotation}.png",
                )
        
        print(f"Finished processing sequence {sequence_name} ({counter}/{len(sequences)}).")
        counter += 1

        if args.debug or args.sanity_check:
            print(">>>> Sanity check done for 1 sequence. Breaking loop.")
            break

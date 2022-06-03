"""Runs evaluation on saved predictions of R2D2-like models on HPatches."""
import os
from os.path import join, exists, expanduser
from genericpath import isdir
from glob import glob
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

from lib.r2d2.extract import extract_keypoints_modified
from relfm.utils.paths import REPO_PATH
from relfm.utils.log import print_update, tqdm_iterator
from relfm.utils.visualize import show_images_with_keypoints, set_latex_fonts
from relfm.utils.matching import evaluate_matching_with_rotation, analyze_result
from relfm.inference.r2d2_on_hpatches import configure_save_dir


if __name__ == "__main__":
    set_latex_fonts(show_sample=False)

    # configure inputs
    data_dir = join(REPO_PATH, "data/hpatches-sequences-release/")

    model_ckpt_paths = {
        "R2D2": join(REPO_PATH, "trained_models/r2d2_WASF_N16.pt"),
    #     "R2D2 - $C_{4}$": join(REPO_PATH, "trained_models/epoch_16_test_model.pt"),
    #     "R2D2 - $SO_{2}$ (Ep 4)": join(REPO_PATH, "trained_models/epoch_3_SO2_4x16_1x32_1x64_2x128.pt"),
    #     "R2D2 - $C_{8}$ (Ep 4)": join(REPO_PATH, "trained_models/epoch_3_C8_4x16_1x32_1x64_2x128.pt"),
    #     "R2D2 - $SO_{2}$ (Ep 18)": join(REPO_PATH, "trained_models/epoch_17_SO2_4x16_1x32_1x64_2x128.pt"),
    # #     "R2D2 - $SO_{2}$ (Ep 0)": join(REPO_PATH, "trained_models/epoch_0_SO2_downsamp_4x16_2x32_1x64_1x128.pt"),
    }

    output_dir = join(expanduser("~"), "outputs/rotation-equivariant-lfm")

    assert isdir(data_dir)
    assert isdir(output_dir)
    # assert exists(model_ckpt_path)

    gap_between_rotations=15
    downsize=True
    imsize=300

    ignore_cache = True
    overwrite_cache = False

    ransac = True

    results = dict()

    # load image sequences
    sequences = sorted(glob(join(data_dir, "*")))

    # load rotation values between 0 and 360 degrees
    rotations = np.arange(0, 360 + 1, gap_between_rotations, dtype=int)

    # set a (list of) pixel threshold value across which to evaluate rotation robustness
    thresholds = [3.]

    # set this to true to see intermediate outputs/messages
    verbose = False

    for model_name, model_ckpt_path in model_ckpt_paths.items():

        # save directory
        save_dir = configure_save_dir(output_dir, model_ckpt_path, dataset_name="hpatches")

        metrics_path = join(save_dir, "metrics.pt")
        if exists(metrics_path) and not ignore_cache:
            print_update(f"Loading cached results for {model_name}")
            results[model_name] = torch.load(metrics_path)["MMA"]
            continue
        
        print_update(f"Generating results for {model_name}")

        # computing Mean Matching Accuracy (MMA)
        mma = defaultdict(list)
        counter = 1
        for sequence in sequences:
            # set path to the source image
            img1_path = join(sequence, "1.ppm")
            img1 = Image.open(img1_path)
            if downsize:
                img1 = img1.resize((imsize, imsize))

            # load outputs for source image
            sequence_name = os.path.basename(sequence)
            save_path = join(save_dir, sequence_name, "1.npy")
            img1_outputs = np.load(save_path, allow_pickle=True).item()

            # possible indices of the target images
            img2_indices = np.arange(2, 7)

            # load all target images at once
            img2s = [Image.open(join(sequence, f"{i}.ppm")) for i in img2_indices]
            if downsize:
                img2s = [img2.resize((imsize, imsize)) for img2 in img2s]

            # load all homographies
            # NOTE that this is not needed since we save the apt H within outputs itself

            rotation_grid, img2_indices_grid  = np.meshgrid(rotations, img2_indices)
            rotation_grid, img2_indices_grid = rotation_grid.flatten(), img2_indices_grid.flatten()

            iterator = tqdm_iterator(
                range(len(rotation_grid)),
                desc=f"Evaluating predictions for {sequence_name} ({counter}/{len(sequences)})\t\t",
            )
            for i in iterator:
                rotation, img2_index = rotation_grid[i], img2_indices_grid[i]

                if verbose:
                    print(f"Image {img2_index} with rotation {rotation}")

                img2 = img2s[img2_index - 2]
                img2_rotated = img2.rotate(rotation)

                save_path = join(save_dir, sequence_name, f"{img2_index}_rotation_{rotation}.npy")
                img2_outputs = np.load(save_path, allow_pickle=True).item()

                # get keypoints and descriptors from the outputs
                kps1 = img1_outputs["keypoints"]
                des1 = img1_outputs["descriptors"]

                kps2 = img2_outputs["keypoints"]
                des2 = img2_outputs["descriptors"]
                H = img2_outputs["H"]

                # show detected keypoints
                if verbose and rotation == 30 and img2_index == 4:
                    show_images_with_keypoints([img1, img2_rotated], [kps1, kps2], radius=2)

                # perform matching
                width, height = img2.size
                result = evaluate_matching_with_rotation(
                    kp1=kps1,
                    des1=des1,
                    kp2=kps2,
                    des2=des2,
                    H=H,
                    width=width,
                    height=height,
                    rotation=rotation,
                    return_metadata=True,
                    threshold=3.,
                    ransac=ransac,
                )

                # show matching results
                if verbose and rotation == 30 and img2_index == 4:
                    analyze_result(img1, img2_rotated, result, K=10, radius=5)

                # compute accuracy across various thresholds
                _match_accu = []
                for threshold in thresholds:
                    _match_accu.append(np.mean(result["distances"] < threshold))

                mma[rotation].append(np.mean(_match_accu))

            counter += 1

        # compute the mean matching accuracy (MMA) for every rotation value
        mma_avg = {k:np.array(mma[k]).mean() for k in mma}
        
        if overwrite_cache:
            # save metrics
            metrics = {
                "MMA": mma_avg,
            }
            torch.save(metrics, metrics_path)
        
        # collect results
        results[model_name] = mma_avg

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.grid(alpha=0.5)
    # ax.set_ylim((0., 1.))

    ransac_suffix = " (RANSAC)" if ransac else ""
    ax.set_title(f"Rotation-equivariance on HPatches dataset {ransac_suffix}", fontsize=23)
    ax.set_xlabel("Rotation angle ", fontsize=18)
    ax.set_ylabel("Mean matching accuracy (MMA)", fontsize=18)

    for model_name, mma_avg in results.items():

        ax.plot(
            list(mma_avg.keys()),
            list(mma_avg.values()),
            "--o",
            label=model_name,
            markersize=8,
            linewidth=2.,
        )
        ax.set_xticks(list(mma_avg.keys()))

    ax.legend(fontsize=17, bbox_to_anchor=(1.25, 0.95), title="Method",)
    fig_dir = "./Figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(join(fig_dir, "final_mma_hpatches_v1.0.pdf"))

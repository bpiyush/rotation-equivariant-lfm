"""
Runs evaluation on saved predictions of given R2D2-like models on HPatches.
"""
import os
from os.path import join, exists, expanduser
from genericpath import isdir
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import time

from lib.r2d2.extract import extract_keypoints_modified
from relfm.utils.paths import REPO_PATH
from relfm.utils.log import print_update, tqdm_iterator
from relfm.utils.visualize import (
    show_images_with_keypoints,
    check_kps_with_homography,
    set_latex_fonts,
    get_colors,
    get_concat_h,
)
from relfm.utils.matching import (
    evaluate_matching_with_rotation,
    analyze_result,
)
from relfm.utils.geometry import (
    append_rotation_to_homography,
    apply_homography_to_keypoints,
    resize, apply_clean_rotation,
)
from relfm.inference.r2d2_on_hpatches import configure_save_dir


def plot_qualitative_results(
        model_ckpt_paths,
        source_images_with_kps,
        target_images_with_kps,
        matched_images,
        sequence_name,
        rotation,
        save_path=None,
        save=True,
        show=False,
    ):
    """Plots qualitative results of matching for a particular sequence at a given rotation."""
    fig, axes = plt.subplots(len(model_ckpt_paths), 1, figsize=(20, 14))

    for i, model_name in enumerate(list(model_ckpt_paths.keys())):
        ax = axes[i]
        source = Image.fromarray(source_images_with_kps[model_name])
        target = Image.fromarray(target_images_with_kps[model_name])
        matchd = matched_images[model_name]
        img = get_concat_h(get_concat_h(source, target), matchd)
        ax.imshow(np.asarray(img))
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"{model_name} at {rotation}$^\circ$"
        ax.set_title(title, fontsize=20)
        
    fig.tight_layout()
    if save:
        save_dir = "./Figures/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = join(save_dir, f"qual_results_rotation_{sequence_name}_{rotation}.pdf")
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved qualitative results to {}".format(save_path))

    if show:
        plt.show()


def plot_quantitative_results(results, ransac, save=True, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.grid(alpha=0.5)
    # ax.set_ylim((0., 1.))

    ransac_suffix = " (RANSAC)" if ransac else ""
    ax.set_title(f"Rotation-equivariance on HPatches dataset {ransac_suffix}", fontsize=20)
    ax.set_xlabel("Rotation angle (degrees)", fontsize=17)
    ax.set_ylabel("Mean matching accuracy (MMA)", fontsize=17)

    i = 0
    colors = get_colors(num_colors=len(results), palette="terrain")
    if len(results) == 5:
        colors = ["blue", "gold", "#b355ed", "green", "red"]
        markers = ["o", "D", "^", "X", "s"]
        linestyles = ["solid", "dashed", "dashed", "dashed", "solid"]
        fillstyles = ["full", "none", "none", "none", "full"]
    elif len(results) == 4:
        colors = ["blue", "gold", "green", "red"]
        markers = ["o", "D", "X", "s"]
        linestyles = ["solid", "dashed", "dashed", "solid"]
        fillstyles = ["full", "none", "none", "full"]
    elif len(results) == 3:
        colors = ["blue", "green", "red"]
        markers = ["o", "X", "s"]
        linestyles = ["solid", "dashed", "solid"]
        fillstyles = ["full", "none", "full"]
    elif len(results) == 2:
        colors = ["blue", "red"]
        markers = ["o", "s"]
        linestyles = ["solid", "solid"]
        fillstyles = ["full", "full"]
    elif len(results) == 1:
        colors = ["blue"]
        markers = ["o"]
        linestyles = ["solid"]
        fillstyles = ["full"]
    else:
        raise NotImplementedError(f"Define colors and markers for {len(results)} number of models")

    for model_name, mma_avg in results.items():

        ax.plot(
            list(mma_avg.keys()),
            list(mma_avg.values()),
            label=model_name,
            markersize=8,
            linewidth=2.,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            fillstyle=fillstyles[i],

        )
        i += 1
        ax.set_xticks(list(mma_avg.keys()))

    ax.legend(fontsize=17, bbox_to_anchor=(1., 0.95), title="Method", title_fontsize=18)

    if save:
        fig_dir = "./Figures"
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(join(fig_dir, "final_mma_hpatches.pdf"), bbox_inches='tight')
        print("Saved quantitative results to ./Figures/final_mma_hpatches.pdf")

    if show:
        plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate R2D2-like models on HPatches dataset",
    )
    parser.add_argument(
        '-m','--models',
        nargs='+',
        default=["R2D2", "SO(2)"],
        help='Models to evaluate on.',
    )
    parser.add_argument(
        "--quantitative",
        action="store_true",
        help="Plot quantitative results",
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="Plot qualitative results",
    )
    parser.add_argument(
        "--rotation_to_visualize",
        default=0,
        type=int,
        help="Rotation to visualize qualitative results",
    )
    parser.add_argument(
        "--target_index_to_visualize",
        default=2,
        type=int,
        help="Index of the target image in the sequence to plot qualitative results",
    )
    parser.add_argument(
        "--sequence_to_visualize",
        default="i_castle",
        help="Sequence to visualize qualitative results",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="Ignore cached results",
    )
    args = parser.parse_args()

    assert args.quantitative or args.qualitative, \
        "Please specify at least one of --quantitative or --qualitative"

    set_latex_fonts(show_sample=False)

    # configure inputs
    data_dir = join(REPO_PATH, "data/hpatches-sequences-release/")

    # list of all models
    model_ckpt_paths = {
        "R2D2": join(REPO_PATH, "trained_models/r2d2_WASF_N16.pt"),
        "C-3PO - $C_{4}$": join(
            REPO_PATH,
            "trained_models/finalmodelC4_epoch_5_4x16_1x32_1x64_2x128.pt",
        ),
        "C-3PO - $C_{8}$": join(
            REPO_PATH,
            "trained_models/finalmodelC8_epoch_1_4x16_1x32_1x64_2x128.pt",
        ),
        "C-3PO - $SO(2)$": join(
            REPO_PATH,
            "trained_models/finalmodelSO2_epoch_17_4x16_1x32_1x64_2x128.pt",
        ),
    }
    # filter models that you want to evaluate on
    keys_to_retain = []
    for model_name in args.models:
        for key in model_ckpt_paths.keys():
            if model_name in key:
                keys_to_retain.append(key)
    model_ckpt_paths = {
        k: v for k, v in model_ckpt_paths.items() if k in keys_to_retain
    }
    print()
    print(">>>>>>>>> Evaluating models:", args.models)
    print()
    time.sleep(3)

    output_dir = join(expanduser("~"), "outputs/rotation-equivariant-lfm")

    assert isdir(data_dir), "Data does not exist at {}.".format(data_dir)
    assert isdir(output_dir), "Output directory does not exist."\
        "Are you sure you have run inference script before running this script?"

    #####
    # DO NOT CHANGE THESE ARGUMENTS SINCE THEY ARE SAME AS
    # IN THE INFERENCE SCRIPT THAT GENERATES PREDICTIONS
    gap_between_rotations=15
    downsize=True
    imsize=300

    ignore_cache = args.ignore_cache
    overwrite_cache = False

    ransac = True
    ransac_threshold = 3.

    sanity_check = False
    crop_post_rotation = False
    #####

    results = dict()

    # load image sequences
    sequences = sorted(glob(join(data_dir, "*")))

    # load rotation values between 0 and 360 degrees
    rotations = np.arange(0, 360 + 1, gap_between_rotations, dtype=int)

    # set a (list of) pixel threshold value across which to evaluate rotation robustness
    thresholds = [3.]

    # set this to true to see intermediate outputs/messages
    verbose = False

    qualitative_analysis = args.qualitative
    if qualitative_analysis:
        print(":::: NOTE: Only performing qualitative analysis, not quantitative analysis.")
        ignore_cache = True
        overwrite_cache = False
        verbose = True

    rotation_to_visualize = args.rotation_to_visualize
    img2_index_to_visualize = args.target_index_to_visualize
    sequence_name_to_visualize = args.sequence_to_visualize
    assert sequence_name_to_visualize in os.listdir(data_dir)

    source_images_with_kps = dict()
    target_images_with_kps = dict()
    matched_images = dict()

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

            # store sequence name
            sequence_name = os.path.basename(sequence)

            if sequence_name != sequence_name_to_visualize and qualitative_analysis:
                continue

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
            
            if downsize:
                # downsize the source image to (args.imsize, args.imsize)
                img1_resized, H1_raw = resize(img1_raw, imsize, imsize)

                # downsize the target images to (args.imsize, args.imsize)
                img2s_resized = []
                for j in range(len(img2s_raw)):
                    img, H2s_raw[j] = resize(img2s_raw[j], imsize, imsize)
                    img2s_resized.append(img)
            else:
                img1_resized = img1_raw
                img2s_resized = img2s_raw
                
            # load all homographies
            H1to2s = [np.loadtxt(join(sequence, f"H_1_{i}")) for i in img2_indices]

            rotation_grid, img2_indices_grid  = np.meshgrid(rotations, img2_indices)
            rotation_grid, img2_indices_grid = rotation_grid.flatten(), img2_indices_grid.flatten()
            iterator = tqdm_iterator(
                range(len(rotation_grid)),
                desc=f"Evaluating predictions for {sequence_name} ({counter}/{len(sequences)})\t\t",
            )
            for i in iterator:
                rotation, img2_index = rotation_grid[i], img2_indices_grid[i]

                if qualitative_analysis:
                    if (rotation != rotation_to_visualize) or (img2_index != img2_index_to_visualize):
                        continue

                img1 = img1_resized.copy()

                # index target image
                img2 = img2s_resized[img2_index - 2].copy()

                # load base homography for source and target image
                H1 = H1_raw.copy()
                H2 = H2s_raw[img2_index - 2].copy()

                if crop_post_rotation:
                    # center crop the source image according to the rotation
                    # NOTE: this does not rotate the image, only crops based on rotation
                    _, _, img1_transformed, H1 = apply_clean_rotation(
                        image=img1, degrees=rotation, H=H1
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

                # load outputs for source image
                save_path = join(
                    save_dir, sequence_name, f"1_rotation_{rotation}.npy",
                )
                img1_outputs = np.load(save_path, allow_pickle=True).item()
                
                # load outputs for target image
                save_path = join(
                    save_dir, sequence_name, f"{img2_index}_rotation_{rotation}.npy",
                )
                img2_outputs = np.load(save_path, allow_pickle=True).item()
                
                # load H
                # H1 = img1_outputs["H1"]
                # H2 = img2_outputs["H2"]
                # H = H1to2s[img2_index - 2].copy()
                # H_transformed = H2 @ H @ np.linalg.inv(H1)
                
                # get keypoints and descriptors from the outputs
                kps1 = img1_outputs["keypoints"]
                des1 = img1_outputs["descriptors"]

                kps2 = img2_outputs["keypoints"]
                des2 = img2_outputs["descriptors"]

                # show detected keypoints
                if verbose and rotation == rotation_to_visualize and img2_index == img2_index_to_visualize:
                    images_with_kps = show_images_with_keypoints(
                        [img1_transformed, img2_transformed],
                        [kps1, kps2],
                        radius=2,
                        save=True,
                        save_path=join(REPO_PATH, f"sample_images/sanity_check_eval_kps_{rotation}.png"),
                        return_images=True,
                    )
                    source_images_with_kps[model_name] = images_with_kps[0]
                    target_images_with_kps[model_name] = images_with_kps[-1]

                # perform matching
                width, height = img2_transformed.size
                result = evaluate_matching_with_rotation(
                    kp1=kps1,
                    des1=des1,
                    kp2=kps2,
                    des2=des2,
                    H=H_transformed,
                    width=width,
                    height=height,
                    rotation=0,
                    return_metadata=True,
                    threshold=3.,
                    ransac=ransac,
                    ransac_threshold=ransac_threshold,
                )

                # show matching results
                if verbose and rotation == rotation_to_visualize and img2_index == img2_index_to_visualize:
                    matched_image = analyze_result(
                        img1_transformed, img2_transformed, result, K=100, radius=5,
                        save=True, save_dir=join(REPO_PATH, "sample_images/"), return_img=True,
                    )
                    matched_images[model_name] = matched_image

                # compute accuracy across various thresholds
                _match_accu = []
                for threshold in thresholds:
                    _match_accu.append(np.mean(result["distances"] < threshold))

                mma[rotation].append(np.mean(_match_accu))

            counter += 1

            if verbose and not qualitative_analysis:
                break

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
    

    if qualitative_analysis:
        print("Finishing qualitative analysis")
        # plot qualitative results
        assert set(source_images_with_kps.keys()) == set(model_ckpt_paths.keys())
        assert set(source_images_with_kps.keys()) == set(target_images_with_kps.keys())
        plot_qualitative_results(
            model_ckpt_paths,
            source_images_with_kps,
            target_images_with_kps,
            matched_images,
            sequence_name=sequence_name_to_visualize,
            rotation=rotation_to_visualize,
        )
    else:
        print("Finishing quantitative analysis")
        # plot results
        plot_quantitative_results(results=results, ransac=ransac)

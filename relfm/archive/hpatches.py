"""Evaluation script for the HPatches dataset."""

from genericpath import isdir
from glob import glob
import numpy as np
from PIL import Image

from lib.r2d2.extract import extract_keypoints_modified
from relfm.utils.log import print_update
from relfm.utils.visualize import show_images_with_keypoints
from relfm.utils.matching import evaluate_matching_with_rotation, analyze_result



if __name__ == "__main__":
    import argparse
    from os.path import join, exists

    from relfm.utils.paths import REPO_PATH


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
    args = parser.parse_args()

    assert isdir(args.data_dir), \
        f"Could not find the HPatches dataset at {args.data_dir}."

    sequences = glob(join(args.data_dir, "*"))
    print_update(f"Evaluating on {len(sequences)} sequences.")
    for sequence in sequences:
        # set path to the source image
        img1_path = join(sequence, "1.ppm")
        img1 = Image.open(img1_path)

        # loop over target images
        for i in range(2, 7):
            img2_path = join(sequence, f"{i}.ppm")
            H_path = join(sequence, f"H_1_{i}")

            rotation = 30

            img2 = Image.open(img2_path)
            img2 = img2.rotate(rotation)

            H = np.loadtxt(H_path)

            # extract keypoints and descriptors for both images
            outputs = extract_keypoints_modified([img1, img2], args.model_ckpt_path)

            # get keypoints and descriptors from the outputs
            kps1 = outputs[0]["keypoints"]
            des1 = outputs[0]["descriptors"]

            kps2 = outputs[1]["keypoints"]
            des2 = outputs[1]["descriptors"]

            # show detected keypoints
            show_images_with_keypoints([img1, img2], [kps1, kps2], radius=2)

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
                rotation=0,
                return_metadata=True,
                threshold=300.,
            )

            analyze_result(img1, img2, result)

            import ipdb; ipdb.set_trace()
            
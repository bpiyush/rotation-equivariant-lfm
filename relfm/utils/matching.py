"""Matching helper functions."""
import cv2
import numpy as np
import torch
from PIL import Image

from relfm.utils.geometry import (
    append_rotation_to_homography,
    apply_homography_to_keypoints,
)
from relfm.utils.visualize import draw_kps_on_image, COLORS, get_concat_h, show_single_image


def add_kps_to_image(img: np.ndarray, kps: np.ndarray):
    
    N = kps.shape[0]
    if kps.shape[1] == 2:
        kps = np.concatenate(kps, np.ones((N, 1)), axis=1)
    assert kps.shape == (N, 3)

    kps = [cv2.KeyPoint(*coord) for coord in kps]
    img_with_kps = cv2.drawKeypoints(img, kps, np.array([]), (255,0,0))
    
    return img_with_kps, kps


def mnn_matcher(descriptors_a, descriptors_b):
    """
    Borrowed from: D2Net's HPatches benchmark notebook.
    Link: https://github.com/mihaidusmanu/d2-net/blob/master/hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb
    """
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    dist = 1- sim[[matches[0], matches[1]]]
    return dist.data.cpu().numpy(), matches.t().data.cpu().numpy()


def find_matches(des1: np.ndarray, des2: np.ndarray):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # compute pairwise distances
    dist, _matches = mnn_matcher(torch.from_numpy(des1), torch.from_numpy(des2))

    # sort based on distances
    order = np.argsort(dist)
    _matches = _matches[order]
    dist = dist[order]
    
    # create cv2.DMatch objects
    matches = []
    for (x, y), d in zip(_matches, np.round(dist, 1)):
        matches.append(cv2.DMatch(_queryIdx=x, _trainIdx=y, _distance=d))
    
    return matches


def add_kps_to_image(img: np.ndarray, kps: np.ndarray, color=(255,0,0)):
    
    N = kps.shape[0]
    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((N, 1))], axis=1)
    assert kps.shape == (N, 3)

    kps = [cv2.KeyPoint(*coord) for coord in kps]
    img_with_kps = cv2.drawKeypoints(img, kps, np.array([]), color)
    
    return img_with_kps, kps



def mnn_matcher_from_D2Net(descriptors_a, descriptors_b):
    """
    Borrowed from: D2Net's HPatches benchmark notebook.
    Link: https://github.com/mihaidusmanu/d2-net/blob/master/hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb
    """
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def evaluate_matching_with_rotation(
        # img1: Image.Image,
        kp1: np.ndarray,
        des1: np.ndarray,
        # img2: Image.Image,
        kp2: np.ndarray,
        des2: np.ndarray,
        H: np.ndarray,
        width: int,
        height: int,
        rotation: int = 0,
        return_metadata: bool = False,
        threshold: float = 5.,
    ):
    """
    Evaluate the matching between two sets of keypoints
    and descriptors for given pair of images.

    Args:
        # img1 (Image.Image): First image.
        kp1 (np.ndarray): First set of keypoints.
        des1 (np.ndarray): First set of descriptors.
        # img2 (Image.Image): Second image.
        kp2 (np.ndarray): Second set of keypoints.
        des2 (np.ndarray): Second set of descriptors.
        H (np.ndarray): Homography matrix (ground truth) transformation between img1 and img2.
        width (int): Width of the image.
        height (int): Height of the image.
        rotation (int): Rotation angle in degrees, default = 0.
        return_metadata (bool): If True, return the metadata of the matching.
        threshold (float): threshold (pixels) to decide an acceptable match, default = 1.
    
    Returns:
        (dict): Dictionary with the all relevant data/metrics.
    """
    # find matches between keypoints
    matches = mnn_matcher_from_D2Net(torch.from_numpy(des1), torch.from_numpy(des2))

    # keep only the matches subset of keypoints
    kp1 = kp1[matches[:, 0], :2]
    kp2 = kp2[matches[:, 1], :2]

    # add rotation to H
    H_combined = append_rotation_to_homography(H, rotation, width, height)
    # H_combined = H

    # project kp1 onto image 2 using homography
    kp2_ground_truth = apply_homography_to_keypoints(kp1, H_combined)

    # compute the distances
    dist = np.sqrt(np.sum((kp2 - kp2_ground_truth) ** 2, axis=1))
    matching_accuracy = np.mean(dist < threshold) if len(dist) > 0 else 0
    good_match_flag = np.array(dist < threshold).astype(int)

    result = {
        "matching_accuracy": matching_accuracy,
    }
    if return_metadata:
        result.update(
            {
                "kp1_matched": kp1,
                "kp2_matched": kp2,
                "kp2_ground_truth": kp2_ground_truth,
                "distances": dist,
                "matches": matches,
                "good_match_flag": good_match_flag,
                "rotation": rotation,
            }
        )
    
    return result


def analyze_result(img1: Image.Image, img2: Image.Image, result, match_thickness: int = 1, K=100, radius=1, model_name=""):
    """Visualizes matching result for given pair of images."""

    K = min(K, result["matches"].shape[0])

    width, height = img1.size

    rotation = result['rotation']

    kp1_matched = result["kp1_matched"]
    kp2_matched = result["kp2_matched"]
    kp2_gt = result["kp2_ground_truth"]

    # add keypoints to images
    img1_with_kps = draw_kps_on_image(
        img1, kp1_matched, color=COLORS["blue"], radius=radius, return_as="PIL",
    )
    img2_with_kps = draw_kps_on_image(
        img2, kp2_matched, color=COLORS["blue"], radius=radius, return_as="PIL",
    )
    img2_with_kps = draw_kps_on_image(
        img2_with_kps, kp2_gt, color=COLORS["yellow"], radius=radius, return_as="PIL",
    )

    # draw green lines for correct matches and red for incorrect matches
    def get_match_color(flag):
        if flag:
            return COLORS["green"]
        else:
            return COLORS["red"]

    match_colors = [get_match_color(x) for x in result["good_match_flag"]]

    img = get_concat_h(img1_with_kps, img2_with_kps)
    selected_match_indices = np.random.choice(np.arange(len(kp1_matched)), K, replace=False)
    for i in selected_match_indices:
        (x1, y1) = int(kp1_matched[i][0]), int(kp1_matched[i][1])
        (x2, y2) = width + int(kp2_matched[i][0]), int(kp2_matched[i][1])

        img = cv2.line(np.asarray(img), (x1, y1), (x2, y2), match_colors[i], thickness=match_thickness)
        img = Image.fromarray(img)
    
    # show the final image
    accuracy = result['matching_accuracy']
    title = f"Matching accuracy: {accuracy:.2f} with rotation {rotation} for {model_name}"
    show_single_image(
        img, title=title, figsize=(10, 8),
    )
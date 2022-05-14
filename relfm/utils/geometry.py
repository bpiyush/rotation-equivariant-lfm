"""Geometric utils."""
import numpy as np
from PIL import Image
from typing import List, Union



def get_image_corners(image: Union[Image.Image, np.ndarray]):
    if isinstance(image, np.ndarray):
        H, W = image.shape[:2]
    elif isinstance(image, Image.Image):
        W, H = image.size
    else:
        raise TypeError("Image has unknown type.")
    
    # # Note: in (y, x) format
    # corners = [[0, 0], [H, 0], [0, W], [H, W]]
    # Note: in (x, y) format
    corners = [[0, 0], [0, H], [W, 0], [W, H]]
    corners = np.array(corners)
    
    return corners


def apply_homography_to_keypoints(kps: np.ndarray, homography: np.ndarray):
    """Applies H to keypoints."""
    
    N = kps.shape[0]
    if kps.shape[1] == 3:
        pos_a = kps[:, :2]
    elif kps.shape[1] == 2:
        pos_a = kps
    else:
        raise ValueError("Invalid shape for kps.")
    
    # homogenize
    pos_a_h = np.concatenate([pos_a, np.ones([N, 1])], axis=1)
    
    # apply homography
    pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
    
    # back to 2D coordinates
    pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]

    return pos_b_proj


def get_2D_rotation_matrix(rotation_in_deg):
    """Returns 2D anticlockwise rotation matrix for given rotation in degrees."""
    rotation_in_rad = np.radians(rotation_in_deg)
    rotation_in_rad = -rotation_in_rad
    R = np.array([
        [np.cos(rotation_in_rad), -np.sin(rotation_in_rad),],
        [np.sin(rotation_in_rad), np.cos(rotation_in_rad)],
    ])
    return R
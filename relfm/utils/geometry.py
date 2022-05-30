"""Geometric utils."""
import math
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


def append_rotation_to_homography(H, rotation, width, height):
    """
    Appends (anticlockwise) rotation to homography.

    Args:
        H (np.ndarray): Homography matrix.
        rotation (int): Rotation angle in degrees.
        width (int): (Target) Image width.
        height (int): (Target) Image height.
    """

    # define the coordinates of the center of the image
    cx, cy = width / 2., height / 2.

    # define translation matrix to move origin to the center of the image
    T_topleft_to_center = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1],
    ])

    # define the rotation matrix
    R2D = get_2D_rotation_matrix(rotation)
    R = np.eye(3)
    R[:2, :2] = R2D

    # define translation matrix to move origin to the top-left corner
    T_center_to_topleft = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1],
    ])

    # # get 2D rotation matrix
    # R2D = get_2D_rotation_matrix(rotation)

    # # construct homography from given 2D rotation matrix
    # HR = np.eye(3)
    # HR[:2, :2] = R2D

    # # define the center of the image
    # C0 = np.array([width / 2., height / 2.])

    # # add the correction translation factor
    # t = (np.eye(2) - R2D) @ C0
    # HR[:2, 2] = t

    # return the final composed homography
    # H_combined = H @ T_topleft_to_center @ R @ T_center_to_topleft
    H_rotation = T_center_to_topleft @ R @ T_topleft_to_center 
    H_combined = H_rotation @ H 
    return H_combined


def center_crop(im, new_width, new_height):
    width, height = im.size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    
    H_crop = np.eye(3)
    H_crop[0, 2] = -left
    H_crop[1, 2] = -top
    
    return im, H_crop


def resize(im, new_width, new_height):
    width, height = im.size
    im = im.resize((new_width, new_height))

    sx = new_width / width
    sy = new_height / height
    H_resize = np.array([
        [sx, 0., 0.],
        [0., sy, 0.],
        [0., 0., 1.],
    ])
    
    return im, H_resize


def apply_clean_rotation(image, degrees, H=np.eye(3)):
    """
    Applies rotation to an image followed by cropping to clean out black areas.
    
    Args:
        image (np.ndarray): input image
        degrees (int): rotation in degrees
        H (np.ndarray): homography matrix, (optional) defaults to $I_{3}$
    
    Returns:
        (tuple): transformed image and homography
    """
    original_width, original_height = image.size
    
    # rotate the image
    rotated = image.rotate(degrees)
    H_rot = append_rotation_to_homography(
        H=H,
        rotation=degrees,
        width=image.size[0],
        height=image.size[1],
    )
    
    # crop rotated image to ignore black areas
    # credits: https://stackoverflow.com/questions/21346670/cropping-rotated-image-with-same-aspect-ratio
    aspect_ratio = float(image.size[0]) / image.size[1]
    rotated_aspect_ratio = float(rotated.size[0]) / rotated.size[1]
    angle = math.fabs(degrees) * math.pi / 180
    if aspect_ratio < 1:
        total_height = float(image.size[0]) / rotated_aspect_ratio
    else:
        total_height = float(image.size[1])
    h = total_height / (aspect_ratio * math.sin(angle) + math.cos(angle))
    w = h * aspect_ratio
    rotated_cropped, H_crop = center_crop(rotated, w, h)
    
    # resize rotated_cropped image to original dimensions
    # credits: https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    rotated_cropped_resized, H_resize = resize(rotated_cropped, image.size[0], image.size[1])
    
    # combine the homographies
    H_combined = H_resize @ H_crop @ H_rot

    image_cropped, _  = center_crop(image, rotated_cropped.size[0], rotated_cropped.size[1])
    image_cropped_resized, _  = resize(image_cropped, image.size[0], image.size[1])
    H_without_rot = H_resize @ H_crop @ H
    
    return rotated_cropped_resized, H_combined, image_cropped_resized, H_without_rot

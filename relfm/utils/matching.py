"""Matching helper functions."""
import cv2
import numpy as np
import torch


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
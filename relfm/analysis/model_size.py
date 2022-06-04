"""Checks number of params (size in MB) for a given model."""
import os
from os.path import join, exists, expanduser
from genericpath import isdir
from glob import glob
import numpy as np
from PIL import Image
import torch

from lib.r2d2.nets.patchnet import Quad_L2Net_ConfCFS
from lib.r2d2.nets.patchnet_equivariant import (
    Discrete_Quad_L2Net_ConfCFS,
    Steerable_Quad_L2Net_ConfCFS,
)


if __name__ == "__main__":
    # from torchsummary import summary

    model_names = []
    model_sizes = []

    print(">>> Checking for R2D2")
    model = Quad_L2Net_ConfCFS()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters for {model.__class__.__name__ }: {pytorch_total_params}")
    model_names.append(model.__class__.__name__)
    model_sizes.append(pytorch_total_params)
    print()

    print(">>> Checking for C-3PO (C4)")
    model = Discrete_Quad_L2Net_ConfCFS(num_rotations=4)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters for {model.__class__.__name__ } (C4): {pytorch_total_params}")
    model_names.append(model.__class__.__name__ + " (C4)")
    model_sizes.append(pytorch_total_params)

    print(">>> Checking for C-3PO (C8)")
    model = Discrete_Quad_L2Net_ConfCFS(num_rotations=8)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters for {model.__class__.__name__ } (C8): {pytorch_total_params}")
    model_names.append(model.__class__.__name__ + " (C8)")
    model_sizes.append(pytorch_total_params)


    print(">>> Checking for C-3PO (SO(2))")
    model = Steerable_Quad_L2Net_ConfCFS(fourier=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters for {model.__class__.__name__ } (SO2): {pytorch_total_params}")
    model_names.append(model.__class__.__name__ + " (SO(2))")
    model_sizes.append(pytorch_total_params)

    print("-" * 100)
    print("Summary of model sizes:")
    for i, name in enumerate(model_names):
        print(name, ":", model_sizes[i])
    print("-" * 100)


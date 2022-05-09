# Rotation-equivariant Local Feature Matching
Rotation equivariance meets local feature matching

## Installation

Follow steps in [here](./setup/README.md) to install the packages in a `conda` environment. You can check if the packages are installed correctly by running:
```bash
python setup/check_packages.py
```
<!-- 
## Datasets

Follow steps in [here](./data/README.md) to download and prepare the datasets. A dataset summary table is provided below. -->

## Getting started

### Training R2D2 from scratch

1. Activate the environment:
    ```bash
    # activate the environment
    conda activate relfm

    # set the python path
    export PYTHONPATH=$PWD/lib/r2d2/:$PWD
    ```
2. Download Aachen dataset: here, you can pass the path to the root data folder. For e.g, `~/datasets/`.
    ```bash
    bash download_aachen.sh -d /path/to/root/data/folder/
    ```
    This will download all required datasets to `/path/to/root/data/folder/`. Note that we symlink this root data folder
    to `$PWD/data/` through the script, i.e., you can checkout data folder directly from `$PWD/data/`.
3. Download R2D2 repository:
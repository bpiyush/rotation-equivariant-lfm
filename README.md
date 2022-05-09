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

1. Refer to the previous section for installation instructions. Activate the environment:
    ```bash
    # activate the environment
    conda activate relfm-v1.0

    # set the python path
    export PYTHONPATH=$PWD/lib/r2d2/:$PWD
    ```
2. Download Aachen dataset: here, you can pass the path to the root data folder. For e.g, `~/datasets/`.
    ```bash
    bash download_aachen.sh -d /path/to/root/data/folder/
    ```
    This will download all required datasets to `/path/to/root/data/folder/`. Note that we symlink this root data folder
    to `$PWD/data/` through the script, i.e., you can checkout data folder directly from `$PWD/data/`.
3. Note that the repo comes with R2D2 code in `lib/r2d2`. So no need to download the code.
4. Training R2D2. Modify the `jobscripts/r2d2_training.job` file to run on your machine. For e.g., you can change the save path   `--save-path /home/lcur1325/models/r2d2-sample/model.pt` as per your preference. You can check the status of your job via the slurm output file.
    ```bash
    sbatch jobscripts/r2d2_training.job
    ```
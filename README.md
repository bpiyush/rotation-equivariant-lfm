# Rotation-equivariant Local Feature Matching
Rotation equivariance meets local feature matching

## Installation

First, clone the repo:
```bash
git clone git@github.com:bpiyush/rotation-equivariant-lfm.git
```

Next, follow steps in [here](./setup/README.md) to install the packages in a `conda` environment.
<!-- You can check if the packages are installed correctly by running:
```bash
python setup/check_packages.py
``` -->
<!-- 
## Datasets

Follow steps in [here](./data/README.md) to download and prepare the datasets. A dataset summary table is provided below. -->


> Tip: To use the conda environment on the login node, you will need to run the following commands before activating the environment:

```bash
module purge
module load 2021
module load Anaconda3/2021.05
conda activate relfm-v1.0
```


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
4. Training R2D2: Run the following command:
    ```bash
    sbatch jobscripts/r2d2_training.job
    ```
    You can check the progress of your job via the slurm output file. You check job status via `squeue | grep $USER`.
    Note that this is only a sample run and will save a model at `/home/$USER/models/r2d2-sample/model.pt`. More customized runs coming soon!



## Evaluation pipeline

To evaluate an R2D2-like model, we use the following evaluation steps:

### Generate and save predictions
Generate keypoint predictions for the HPatches dataset and save them to disk.
You need to pass the data director, base output folder and the model checkpoint path.
```sh
(relfm-1.0) $ python relfm/inference/r2d2_on_hpatches.py \
    --data_dir ./data/hpatches-sequences-release/ \
    --output_dir ~/outputs/rotation-equivariant-lfm/ \
    --model_ckpt_path ./checkpoints/r2d2_WASF_N16.pt \
    --num_keypoints 1000 
```
This will run inference, generate outputs and save them to the folder:
`~/outputs/rotation-equivariant-lfm/hpatches/r2d2_WASF_N16`. Depending on your checkpoint name, it will create a new folder for a new checkpoint.

The output shall have 1 folder per image sequence in HPatches, for e.g., `v_home`. Each folder shall contain the following files:

* `1.npy`: the predicted keypoint locations and descriptor vectors for the source image.
* `t_rotation_R.npy`: the predicted keypoint locations and descriptor vectors for the target image with index `t` and rotation `R`, $\forall t \in \{2, 3, 4, 5, 6\}, R \in \{0, 15, 30, .., 345, 360\}$.

Tips:
* Please run the following to see all options:
    ```sh
    (relfm-1.0) $ python relfm/inference/r2d2_on_hpatches.py --help
    ```
* If you want to debug and run this script only for 1 image sequence, let's say, `v_home`, then run:
    ```sh
    (relfm-1.0) $ python relfm/inference/r2d2_on_hpatches.py \
        --data_dir ./data/hpatches-sequences-release/ \
        --output_dir ~/outputs/rotation-equivariant-lfm/ \
        --model_ckpt_path ./checkpoints/r2d2_WASF_N16.pt \
        --num_keypoints 1000 \
        --seq_prefix v_home \
        --debug
    ```

### Evaluate predictions
Use the generated predictions to evaluation robustness of feature matching to rotations. You can do this by running [this notebook](./notebooks/eval_on_hpatches.ipynb).
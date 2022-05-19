## Installation

We use `conda` environment manager to install the packages. Depending on your machine, you may create a new environment.

### CPU Version

For the CPU version, we currently do not have an automated installation script. You can install the packages manually by running the following commands:

```bash
conda create -y -n relfm-v1.0 python=3.9
conda activate relfm-v1.0
conda install -y tqdm pillow numpy matplotlib scipy
pip install ipdb ipython jupyter jupyterlab gdown opencv-python termcolor natsort
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
```

> :warning: **Note**: This has been tested on Mac M1 machine.

### GPU Version

<!-- > **Important**: I noticed that on Lisa, you need to use job script to create an environment. Environment created on your login node probably would not work. Once the environment is created, you can activate it and install other packages. Use the following command to create an environment: -->

Run the following command to create an environment:
```bash
sbatch jobscripts/create_gpu_env.job
```

Once it is created, you can activate it and install other packages.

To activate the environment, run the following command (on your login node):
```sh
module load 2021
module load Anaconda3/2021.05
conda activate relfm-v1.0
```

To check if the packages are installed correctly, run the following command:
```sh
python setup/check_packages.py
```

<!-- Once the environment `relfm-v1.0` is created, activate it and install other packages.
Please follow the same instructions as above except use the apt CUDA version while installing `torch`.

For e.g., using CUDA 10.1, use:
```bash
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
 -->

## Updating the environment

Suppose you want to install an additional package, say `termcolor`, you need to add it to the jobscript and run it again.
```bash
(old) pip install .....some packages.....
(new) pip install .....some packages..... termcolor
```
Then, run the jobscript again.
```bash
sbatch jobscripts/create_gpu_env.job
```


> Tip: To use the conda environment on the login node, you will need to run the following commands before activating the environment:

```bash
module purge
module load 2021
module load Anaconda3/2021.05
conda activate relfm-v1.0
```

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
Use the generated predictions to evaluation robustness of feature matching to rotations. You can do this by running the following notebook.
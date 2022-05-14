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

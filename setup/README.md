## Installation

### CPU Version

```bash
conda create -y -n relfm python=3.9
conda activate relfm
conda install -y python tqdm pillow numpy matplotlib scipy
pip install ipdb ipython jupyter jupyterlab
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0

# install kapture to manage datasets
# (note that llvmlite and numba can only be installed via conda-forge on mac M1)
conda install llvmlite
conda install numba
pip install kapture
```

> :warning: **Note**: This has been tested on Mac M1 machine.

### GPU Version

Please follow the same instructions except use the apt CUDA version while installing `torch`.


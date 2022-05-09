## Installation

### CPU Version

```bash
conda create -y -n relfm python=3.9
conda activate relfm
conda install -y tqdm pillow numpy matplotlib scipy
pip install ipdb ipython jupyter jupyterlab gdown opencv-python
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
```

> :warning: **Note**: This has been tested on Mac M1 machine.

### GPU Version

Please follow the same instructions except use the apt CUDA version while installing `torch`.

For e.g., using CUDA 10.1, use:
```bash
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

On Lisa, please use the GPU version.

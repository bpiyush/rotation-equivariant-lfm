
## R2D2

* Adds code to save checkpoint at every epoch (`r2d2/train.py`)
* Modified code to allow users to continue training from a model checkpoint. The original code used a function that gave PyTorch-errors.
* Adds a function `extract_keypoints_modified` in `r2d2/extract.py` that produces outputs for a list of images, given a model (or checkpoint path)
* Adds a file `lib/r2d2/nets/patchnet_equivariant.py` as our steerable R2D2 model file.


## Aachen Dataset (and variants for R2D2 training)

1. Download the datasets (original tutorial [here](https://github.com/naver/r2d2#training-the-model))
```sh
bash download_aachen.sh -d /path/to/root/data/folder/ # e.g. ~/datasets/
```

2. Download R2D2 repository
```sh
mkdir lib/
cd lib/
git clone git@github.com:naver/r2d2.git
cd -
```

3. Check sample data demo:
```
export PYTHONPATH=./lib/r2d2/
python -m lib.r2d2.tools.dataloader "PairLoader(aachen_flow_pairs)"
```
This may not work on Lisa or a remote server with no GUI.

4. Training R2D2
```sh
sbatch jobscripts/r2d2_training.job
```
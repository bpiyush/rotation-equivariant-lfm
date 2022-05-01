## Data Preparation

This section has instructions for setting up the required datasets. Roughly,
the following steps are performed:
1. Downloading the dataset
2. Converting the dataset to the required format
3. Visualizing samples from the dataset

### Aachen Day Night v1.1

**Download**: Refer to [this tutorial](https://github.com/naver/kapture/blob/main/doc/tutorial.adoc#download-a-dataset) on Kapture for managing this dataset.

```bash
# activate the conda environment
conda activate relfm

# navigate to the directory where you want to download the dataset
# example: ~/datasets/kapture_datasets/
cd /path/to/dataset

# update the list from repositories
kapture_download_dataset.py update

# display the list of datasets: filter out relevant ones
kapture_download_dataset.py list | grep Aachen-Day-Night

# download relevant subsets of the dataset

# Memory requirement: about 5.54 GBs
kapture_download_dataset.py install Aachen-Day-Night-v1.1_mapping

# Memory requirement: about 0.5 GBs in total
kapture_download_dataset.py install Aachen-Day-Night-v1.1_query_day
kapture_download_dataset.py install Aachen-Day-Night-v1.1_query_night
```

### HPatches Dataset

The official evaluation code is provided [here](https://github.com/hpatches/hpatches-dataset). Their evaluation scripts automatically download the dataset and convert it to the required format.

However, since we need raw images, we need to download the dataset manually.
```bash
# example: ~/datasets/
cd /path/to/dataset/
# wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-release.tar.gz
http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
# tar -xvf hpatches-release.tar.gz
tar -xvf hpatches-sequences-release.tar.gz
# rm -rf hpatches-release.tar.gz
rm -rf hpatches-sequences-release.tar.gz
```

**Visualize**: Check out sample images from the dataset.
```bash
(relfm) $ ipython
```
```python
In [1]: %matplotlib inline
In [2]: from PIL import Image
In [3]: path = "~/datasets/hpatches-sequences-release/v_yard/1.ppm"
In [4]: img = Image.open(path)
In [5]: img.show()
```
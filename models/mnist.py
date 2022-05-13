import torch
from torch.utils.data import Dataset
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from PIL import Image
import numpy as np


class MnistDataset(Dataset):
    """MNIST Dataset - class borrowed from Notebook by Gabrielle"""
    def __init__(self, mode, rotated: bool = True):
        assert mode in ['train', 'test']

        if mode == "train":
            file = "mnist/mnist_train.amat"
        else:
            file = "mnist/mnist_test.amat"

        data = np.loadtxt(file)

        images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)

        # images are padded to have shape 29x29.
        # this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
        pad = Pad((0, 0, 1, 1), fill=0)

        # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
        # we upsample an image by a factor of 3, rotate it and finally downsample it again
        resize1 = Resize(87) # to upsample
        resize2 = Resize(29) # to downsample

        totensor = ToTensor()

        if rotated:
            self.images = torch.empty((images.shape[0], 1, 29, 29))
            for i in tqdm(range(images.shape[0]), leave=False):
                img = images[i]
                img = Image.fromarray(img, mode='F')
                r = (np.random.rand() * 360.)
                self.images[i] = totensor(resize2(resize1(pad(img)).rotate(r, Image.BILINEAR))).reshape(1, 29, 29)
        else:
            self.images = torch.zeros((images.shape[0], 1, 29, 29))
            self.images[:, :, :28, :28] = torch.tensor(images).reshape(-1, 1, 28, 28)

        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        return image, label

    def __len__(self):
        return len(self.labels)


def generate_loaders(batch_size=64):
    mnist_data = MnistDataset(mode='train', rotated=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size)

    return mnist_loader

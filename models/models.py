import torch
from tqdm import tqdm
import numpy as np
from escnn import *
from mnist import MnistDataset, test_model_single_image


class CNSteerableL2(torch.nn.Module):
    def __init__(self, n_rotations=4, n_hidden=64, n_classes=10):
        super(CNSteerableL2, self).__init__()
        self.r2_act = gspaces.rot2dOnR2(N=n_rotations)
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.mask = nn.MaskModule(in_type, 29, margin=1)

        activation1 = nn.ELU(nn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation1.in_type
        self.block = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation1,
        )

        self.pool = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        output_invariant_type = nn.FieldType(self.r2_act, n_hidden * [self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x):
        x = self.input_type(input)
        x = self.mask(x)

        # Convolutions
        x = self.block(x)
        x = self.pool(x)

        # Final pool
        x = self.invariant_map(x)
        x = x.tensor

        # Prediction
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x


class CNSteerableCNN(torch.nn.Module):
    def __init__(self, n_rotations=4, n_classes=10):
        super(CNSteerableCNN, self).__init__()

        # the model is equivariant to rotations by multiples of 2pi/N
        self.r2_act = gspaces.rot2dOnR2(N=n_rotations)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # We need to mask the input image since the corners are moved outside the grid under rotations
        self.mask = nn.MaskModule(in_type, 29, margin=1) # Todo: Whattefuck?

        # convolution 1
        # first we build the non-linear layer, which also constructs the right feature type
        # we choose 8 feature fields, each transforming under the regular representation of C_4
        activation1 = nn.ELU(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]), inplace=True)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation1,
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 16 regular feature fields
        activation2 = nn.ELU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation2
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 32 regular feature fields
        activation3 = nn.ELU(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]), inplace=True)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation3
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 32 regular feature fields
        activation4 = nn.ELU(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]), inplace=True)
        out_type = activation4.in_type
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation4
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 64 regular feature fields
        activation5 = nn.ELU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True)
        out_type = activation5.in_type
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation5
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation6 = nn.ELU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True)
        out_type = activation6.in_type
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.IIDBatchNorm2d(out_type),
            activation6
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        # number of output invariant channels
        c = 64

        output_invariant_type = nn.FieldType(self.r2_act, c*[self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)


        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, n_classes),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = self.input_type(input)

        # mask out the corners of the input image
        x = self.mask(x)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # Each layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # extract invariant features
        x = self.invariant_map(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layer
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x


if __name__ == '__main__':
    # build the rotated training and test datasets

    mnist_train = MnistDataset(mode='train', rotated=True)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)

    mnist_test = MnistDataset(mode='test', rotated=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

    # for testing purpose, we also build a version of the test set with *non*-rotated digits
    raw_mnist_test = MnistDataset(mode='test', rotated=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c4_l2 = CNSteerableL2(n_rotations=4, n_classes=10).to(device)

    # x, y = next(iter(raw_mnist_test))
    # print(type(x))
    # test_model_single_image(c4_l2, x, device, N=24)
    #
    # raise

    # Train
    epochs = 2
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(c4_l2.parameters(), lr=5e-4, weight_decay=1e-8)

    for epoch in tqdm(range(epochs)):
        c4_l2.train()
        for i, (x, label) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.to(device)
            label = label.to(device)
            pred = c4_l2(x)

            loss = loss_function(label, pred)
            loss.backward()

            optimizer.step()



from e2cnn import gspaces
from e2cnn import nn
import torch


class DiscreteRotationCNN(torch.nn.Module):
    def __init__(self, kernel_size, num_rotations, channels_in, channels_hidden, channels_out):
        super(DiscreteRotationCNN, self).__init__()
        self.kernel_size = kernel_size

        # group actions based on discrete rotation group (C_n)
        self.group_actions = gspaces.Rot2dOnR2(num_rotations)

        # feature types + representations
        self.type_in = nn.FieldType(self.group_actions,  channels_in*[self.group_actions.trivial_repr])
        self.type_out = nn.FieldType(self.group_actions,  channels_hidden*[self.group_actions.regular_repr])

        # single convolution
        self.conv = nn.R2Conv(self.type_in, self.type_out, kernel_size=self.kernel_size)
        # relu act_fn
        self.act_fn = nn.ReLU(self.type_out)
        # pooling through conv with kernel size 1
        self.pool_conv = torch.nn.Conv2d(num_rotations * channels_hidden, channels_out, 1)

    def forward(self, x):
        # Make e2cnn object
        x = nn.GeometricTensor(x, self.type_in)     # shape [num_img, num_chan, H, W]
        out = self.conv(x)                          # shape [num_img, num_chan * num_rotations, H, W]
        out = self.act_fn(out)                      # shape [num_img, num_chan * num_rotations, H, W]
        out = self.pool_conv(out.tensor)            # shape [num_img, 1, H, W]

        return out


if __name__ == '__main__':
    discreteRotationCNN = DiscreteRotationCNN(5, 8, 3, 10, 2)
    discreteRotationCNN = DiscreteRotationCNN(
        kernel_size=3,
        num_rotations=5,
        channels_in=3,
        channels_hidden=10,
        channels_out=1
    )

    # Generate random images
    x = torch.randn(100, 3, 32, 32)     # shape [num_img, chan_in, H_in, W_in]
    y = discreteRotationCNN(x)          # shape [num_img, chan_out, H_out, W_out]

    print(x.shape)
    print(y.shape)

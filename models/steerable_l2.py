import torch
from escnn import *


class SteerableL2(torch.nn.Module):
    def __init__(self, r2_act, fourier):
        super(SteerableL2, self).__init__()
        self.r2_act = r2_act
        self.fourier = fourier

        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.mask = nn.MaskModule(in_type, 29, margin=1)

        # Todo: BN weights are not set to 0/1

        # 3x3 conv 32
        activation1 = self.get_act_fn(32)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation1,
        )

        # 3x3 conv 32
        in_type = self.block1.out_type
        activation2 = self.get_act_fn(32)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation2,
        )

        # 3x3 conv 64 /2
        in_type = self.block2.out_type
        activation3 = self.get_act_fn(64)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, dilation=2),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation3,
        )

        # 3x3 conv 64
        in_type = self.block3.out_type
        activation4 = self.get_act_fn(64)
        out_type = activation4.in_type
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation4,
        )

        # 3x3 conv 128 /2
        in_type = self.block4.out_type
        activation5 = self.get_act_fn(128)
        out_type = activation5.in_type
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, dilation=2),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation5,
        )

        # Todo: Add missing normalisation

        # 3x3 conv 128
        in_type = self.block5.out_type
        activation6 = self.get_act_fn(128)
        out_type = activation6.in_type
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation6,
        )

        # 128, k=7, stride=8, not bn or act_fn
        in_type = self.block6.out_type
        out_type = activation6.in_type
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, dilation=8, padding=24),
        )

        output_invariant_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

    def get_act_fn(self, c, freq=4, samples=16):
        if self.fourier:
            return nn.FourierELU(self.r2_act, c, irreps=[(f,) for f in range(freq)], N=samples, inplace=True)
        else:
            return nn.ReLU(nn.FieldType(self.r2_act, c * [self.r2_act.regular_repr]), inplace=True)

    def forward(self, x):
        x = self.input_type(x)
        x = self.mask(x)

        # Convolutions
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # Final pool
        x = self.invariant_map(x)

        # Convert from Geometric Tensor to Tensor
        x = x.tensor
        x = torch.squeeze(x)

        return x

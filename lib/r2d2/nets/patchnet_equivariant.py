import torch
from escnn import *
import argparse
import torch.nn.functional as F
import torchvision

class Steerable_BaseNet (torch.nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = self.softmax( ureliability ))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)


class Steerable_Quad_L2Net(Steerable_BaseNet):
    def __init__(self, r2_act, fourier):
        super(Steerable_Quad_L2Net, self).__init__()
        self.r2_act = r2_act
        self.fourier = fourier

        # in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        # self.input_type = in_type
        # self.mask = nn.MaskModule(in_type, 29, margin=1)

        # TODO: Is this correct? We rewrote this to no longer be MNIST specific
        in_type = nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
        # in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        #TODO: Need maskmodule, and MNIST specific?
        # self.mask = nn.MaskModule(in_type, 29, margin=1)

        # TODO: BN weights are not set to 0/1

        # TODO: r2d2 implementation uses eps 1e-5; was there a reason for the 1e-4?
        eps = 1e-5
        # TODO: Use Affine=False for BN in R2D2?
        affine = False

        # 3x3 conv 32
        activation1 = self.get_act_fn(32)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
            activation1,
        )

        # 3x3 conv 32
        in_type = self.block1.out_type
        activation2 = self.get_act_fn(32)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
            activation2,
        )


        #TODO: This layer was missing in the original version, or was there a reason?
        # 3x3 conv 64 /2
        in_type = self.block2.out_type
        activation_extra = self.get_act_fn(64)
        out_type = activation_extra.in_type
        self.block_extra = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, dilation=1),
            nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
            activation_extra,
        )


        # 3x3 conv 64 /2
        in_type = self.block_extra.out_type
        activation3 = self.get_act_fn(64)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, dilation=2),
            nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
            activation3,
        )


        # TODO
        # # 3x3 conv 64
        # in_type = self.block3.out_type
        # activation4 = self.get_act_fn(64)
        # out_type = activation4.in_type
        # self.block4 = nn.SequentialModule(
        #     nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
        #     nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
        #     activation4,
        # )



        # 3x3 conv 128 /2
        # in_type = self.block4.out_type
        in_type = self.block3.out_type
        activation5 = self.get_act_fn(128)
        out_type = activation5.in_type
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, dilation=2),
            nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
            activation5,
        )


        # Todo: Add missing normalisation

        #TODO: (15) has padding 4 and dilation 4.
        # 3x3 conv 128
        # in_type = self.block5.out_type
        # activation6 = self.get_act_fn(128)
        # out_type = activation6.in_type
        # self.block6 = nn.SequentialModule(
        #     nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
        #     nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
        #     activation6,
        # )

        in_type = self.block5.out_type
        activation6 = self.get_act_fn(128)
        out_type = activation6.in_type
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=4, dilation=4),
            nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
            activation6,
        )


        #TODO: From here it is the 3 2x2 convs:
        #TODO: get_act_fn is only used for out_type.
        # in_type = self.block6.out_type
        # out_type = activation6.in_type
        # self.block7 = nn.SequentialModule(
        #     nn.R2Conv(in_type, out_type, kernel_size=2, padding=2, dilation=4),
        #     # nn.R2Conv(in_type, out_type, kernel_size=7, dilation=8, padding=24),
        #     # nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
        # )
        # in_type = self.block7.out_type
        # activation8 = self.get_act_fn(128)
        # out_type = activation8.in_type
        # self.block8 = nn.SequentialModule(
        #     nn.R2Conv(in_type, out_type, kernel_size=2, padding=4, dilation=8),
        #     nn.IIDBatchNorm2d(out_type, eps=eps, affine=affine),
        # )
        # in_type = self.block8.out_type
        # activation9 = self.get_act_fn(128)
        # out_type = activation9.in_type
        # self.block9 = nn.SequentialModule(
        #     nn.R2Conv(in_type, out_type, kernel_size=2, padding=8, dilation=16),
        # )

        # 128, k=7, stride=8, not bn or act_fn
        in_type = self.block6.out_type
        out_type = activation6.in_type
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=7, dilation=8, padding=24),
        )

        output_invariant_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

        self.out_dim = output_invariant_type.size

    def get_act_fn(self, c, freq=2, samples=8):
        if self.fourier:
            return nn.FourierELU(self.r2_act, c, irreps=[(f,) for f in range(freq)], N=samples, inplace=True)
        else:
            return nn.ReLU(nn.FieldType(self.r2_act, c * [self.r2_act.regular_repr]), inplace=True)

    def forward_equi(self, x):
        # rgb2gray = torchvision.transforms.Grayscale()
        # x = rgb2gray(x)

        x = self.input_type(x)
        # x = self.mask(x)

        # Convolutions
        x = self.block1(x)
        x = self.block2(x)
        x = self.block_extra(x)
        x = self.block3(x)
        # x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # Final pool
        # TODO: not necessary?
        x = self.invariant_map(x)

        # Convert from Geometric Tensor to Tensor
        x = x.tensor
        x = torch.squeeze(x)

        return x


class Steerable_Quad_L2Net_ConfCFS (Steerable_Quad_L2Net):
    def __init__(self, r2_act=None, fourier=True):
        #TODO: This is hardcoded for now, but works for now.
        if not r2_act:
            r2_act = gspaces.rot2dOnR2(N=-1)

        Steerable_Quad_L2Net.__init__(self, r2_act, fourier)
        # reliability classifier
        self.clf = torch.nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = torch.nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x):
        # assert self.ops, "You need to add convolutions first"
        # for op in self.ops:
        #     x = op(x)
        x = self.forward_equi(x)

        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)



def main(args):
    # mnist_loader = generate_loaders(args.batch_size)
    if args.group == 'cn':
        r2_act = gspaces.rot2dOnR2(N=args.num_rotations)
        fourier = False
    elif args.group == 'so2':
        r2_act = gspaces.rot2dOnR2(N=-1)
        fourier = True
    else:
        raise ValueError(f'Do not recognize group {args.group}')

    model = Steerable_Quad_L2Net_ConfCFS(r2_act, fourier=fourier).to(args.device)
    print(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--group', type=str, default='so2',
                        help='group')
    parser.add_argument('--num_rotations', type=int, default=4,
                        help='number of rotations C_n group')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight_decay')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='epochs')

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(parsed_args)

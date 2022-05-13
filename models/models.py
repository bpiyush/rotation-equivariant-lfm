import torch
import argparse
from escnn import *
from mnist import generate_loaders
from steerable_l2 import SteerableL2


def main(args):
    mnist_loader = generate_loaders(args.batch_size)

    if args.group == 'cn':
        r2_act = gspaces.rot2dOnR2(N=args.num_rotations)
        fourier = False
    elif args.group == 'so2':
        r2_act = gspaces.rot2dOnR2(N=-1)
        fourier = True
    else:
        raise ValueError(f'Do not recognize group {args.group}')

    model = SteerableL2(r2_act, fourier=fourier).to(args.device)

    for i, (imgs, label) in enumerate(mnist_loader):
        imgs = imgs.to(args.device)
        features = model(imgs)
        print(f'Images shape: {imgs.shape}')
        print(f'Features shape: {features.shape}')
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--group', type=str, default='cn',
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

import torch
import argparse
from tqdm import tqdm
from escnn import *
from mnist import generate_loaders
from steerable_l2 import SteerableL2
from train import train, test


def main(args):
    train_loader, test_loader = generate_loaders(args.batch_size)
    if args.group == 'cn':
        r2_act = gspaces.rot2dOnR2(N=args.num_rotations)
        fourier = False
    elif args.group == 'so2':
        r2_act = gspaces.rot2dOnR2(N=-1)
        fourier = True
    else:
        raise ValueError(f'Do not recognize group {args.group}')

    model = SteerableL2(r2_act, fourier=fourier).to(args.device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in tqdm(range(args.epochs)):
        model.train()
        train(model, train_loader, loss_function, optimizer, args.device)
        model.eval()
        test_score = test(model, test_loader, args.device)
        print('Intermediate test score:', test_score)

    print(f'Final test score: {test_score}')


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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='epochs')

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(parsed_args)

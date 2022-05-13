import torch
from tqdm import tqdm
import numpy as np
from escnn import *
from mnist import MnistDataset, test_model_single_image
from torch.nn import LocalResponseNorm


class CNSteerableL2(torch.nn.Module):
    def __init__(self, n_rotations=4, n_hidden=64, n_classes=10):
        super(CNSteerableL2, self).__init__()

        self.r2_act = gspaces.rot2dOnR2(N=n_rotations)
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.mask = nn.MaskModule(in_type, 29, margin=1)

        # TODO: 32, 64 shit?
        # TODO: BN weights setten op 0 en 1
        # 1. Block of actions
        activation1 = nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation1,
        )

        # 2. Block of actions
        in_type = self.block1.out_type
        activation2 = nn.ReLU(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation2,
        )

        # 3. Block of actions
        in_type = self.block2.out_type
        activation3 = nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=2),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation3,
        )

        # 4. Block of actions
        in_type = self.block3.out_type
        activation4 = nn.ReLU(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation4.in_type
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation4,
        )

        # 5. Block of actions
        in_type = self.block4.out_type
        activation5 = nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation5.in_type
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=2),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation5,
        )

        # 6. Block of actions
        in_type = self.block5.out_type
        activation6 = nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation6.in_type
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation6,
        )

        # 7. Block of actions
        # TODO: Add LRN instead of activation
        in_type = self.block6.out_type
        # TODO: LRN size???
        # LRN = LocalResponseNorm(256, alpha=256, k=0, beta=0.5),
        activation7 = nn.ReLU(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True)
        out_type = activation6.in_type
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=8),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation7
        #     LRN
        )

        # self.pool1 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        output_invariant_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

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

        # x = self.pool1(x)

        # Final pool
        x = self.invariant_map(x)

        # Convert from Geometric Tensor to Tensor
        x = x.tensor
        x = torch.squeeze(x)

        return x


class SO2SteerableL2(torch.nn.Module):
    def __init__(self, n_hidden=64, n_classes=10):
        super(SO2SteerableL2, self).__init__()

        self.r2_act = gspaces.rot2dOnR2(N=-1)
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.mask = nn.MaskModule(in_type, 29, margin=1)

        # TODO: 32, 64 shit?
        # TODO: BN weights setten op 0 en 1
        # 1. Block of actions
        activation1 = nn.FourierELU(self.r2_act, 32, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation1,
        )

        # 2. Block of actions
        in_type = self.block1.out_type
        activation2 = nn.FourierELU(self.r2_act, 32, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation2,
        )

        # 3. Block of actions
        in_type = self.block2.out_type
        activation3 = nn.FourierELU(self.r2_act, 64, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=2),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation3,
        )

        # 4. Block of actions
        in_type = self.block3.out_type
        activation4 = nn.FourierELU(self.r2_act, 64, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation4.in_type
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation4,
        )

        # 5. Block of actions
        in_type = self.block4.out_type
        activation5 = nn.FourierELU(self.r2_act, 128, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation5.in_type
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=2),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation5,
        )

        # 6. Block of actions
        in_type = self.block5.out_type
        activation6 = nn.FourierELU(self.r2_act, 128, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation6.in_type
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation6,
        )

        # 7. Block of actions
        # TODO: Add LRN instead of activation
        in_type = self.block6.out_type
        # TODO: LRN size???
        # LRN = LocalResponseNorm(256, alpha=256, k=0, beta=0.5),
        activation7 = nn.FourierELU(self.r2_act, 128, irreps=[(f,) for f in range(4)], N=16, inplace=True)
        out_type = activation6.in_type
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=8),
            nn.IIDBatchNorm2d(out_type, eps=1e-4),
            activation7
        )

        # self.pool1 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        output_invariant_type = nn.FieldType(self.r2_act, 128 * [self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

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

        # x = self.pool1(x)

        # Final pool
        x = self.invariant_map(x)

        # Convert from Geometric Tensor to Tensor
        x = x.tensor
        x = torch.squeeze(x)

        return x


def train(model, loader, loss_function, optimizer):
    for i, (x, label) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        label = label.to(device)
        pred = model(x)

        loss = loss_function(pred, label)
        loss.backward()

        optimizer.step()


def test(model, loader):
    # test over the full rotated test set
    total = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(loader):
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()

    return correct/total*100.


if __name__ == '__main__':
    # build the rotated training and test datasets

    mnist_train = MnistDataset(mode='train', rotated=True)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)

    mnist_test = MnistDataset(mode='test', rotated=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNSteerableL2(n_rotations=4, n_classes=10).to(device)
    # model = SO2SteerableL2(n_classes=10).to(device)

    # Train
    epochs = 20
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)

    for epoch in tqdm(range(epochs)):
        model.train()
        train(model, train_loader, loss_function, optimizer)
        model.eval()
        test_score = test(model, test_loader)
        print('Intermediate test score:', test_score)

    # Test
    test_score = test(model, test_loader)
    print(f'Final test score: {test_score}')

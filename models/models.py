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
            torch.nn.Linear(64 * 21 * 21, n_hidden),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(n_hidden, n_classes)
        )

    def forward(self, x):
        x = self.input_type(x)
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
    c4_l2 = CNSteerableL2(n_rotations=4, n_classes=10).to(device)

    # Train
    epochs = 20
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(c4_l2.parameters(), lr=5e-4, weight_decay=1e-8)

    for epoch in tqdm(range(epochs)):
        c4_l2.train()
        train(c4_l2, train_loader, loss_function, optimizer)
        c4_l2.eval()
        test_score = test(c4_l2, test_loader)
        print(test_score)

    test_score = test(c4_l2, test_loader)
    print(f'Final test score: {test_score}')

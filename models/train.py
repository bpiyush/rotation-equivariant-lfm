import torch


def train(model, loader, loss_function, optimizer, device):
    for i, (x, label) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(device)
        label = label.to(device)

        pred = model(x)
        loss = loss_function(pred, label)

        loss.backward()
        optimizer.step()


def test(model, loader, device):
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

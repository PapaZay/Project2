import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, input_channels=4):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=96, kernel_size=11, stride=4)
        self.local_response1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.local_response2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=(256 * 5), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.local_response1(self.conv1(x))))
        x = self.pool2(F.relu(self.local_response2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                data.append(list(map(float, line.strip().split())))
            except ValueError:
                print("Error converting line to float:", line)
    return np.array(data)

train_data = read_data('train.txt')
train_labels = train_data[:, 0]
train_data = train_data[:, 1:]

validation_data = read_data('validation.txt')
validation_labels = validation_data[:, 0]
validation_data = validation_data[:, 1:]

test_data = read_data('test.txt')
test_labels = test_data[:, 0]
test_data = test_data[:, 1:]


train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

validation_data = torch.tensor(validation_data, dtype=torch.float32)
validation_labels = torch.tensor(validation_labels, dtype=torch.long)

test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)


BATCH_SIZE = 512


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels),
                                           batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(validation_data, validation_labels),
                                                batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_labels),
                                          batch_size=BATCH_SIZE, shuffle=True)


model = AlexNet(input_channels=4).to(device)


optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    print("inside train")
    model.train()
    for batch_ids, (img, classes) in enumerate(train_loader):
        classes = classes.type(torch.LongTensor)
        img, classes = img.to(device), classes.to(device)
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, classes)
        loss.backward()
        optimizer.step()
        if (batch_ids + 1) % 2 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_ids * len(img), len(train_loader.dataset),
                       100. * batch_ids / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, classes in test_loader:
            img, classes = img.to(device), classes.to(device)
            y_hat = model(img)
            test_loss += F.nll_loss(y_hat, classes, reduction='sum').item()
            _, y_pred = torch.max(y_hat, 1)
            correct += (y_pred == classes).sum().item()
        test_loss /= len(test_loader.dataset)
        print("\n Test set: Average loss: {:.0f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        print('=' * 30)


if __name__ == '__main__':
    seed = 42
    EPOCHS = 2

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


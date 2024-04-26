import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x


def one_hot_encoder(sequence):
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoding_matrix = np.zeros((len(sequence), len(nucleotide_map)), dtype=int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide in nucleotide_map:
            encoding_matrix[i, nucleotide_map[nucleotide]] = 1
    return encoding_matrix


def read_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        header = file.readline()  
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue  
            label, sequence = parts
            try:
                label = int(label)
            except ValueError:
                continue  
            sequences.append((label, sequence))
    return sequences



class CustomDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        label, sequence = self.sequences[idx]
        
        one_hot_sequence = one_hot_encoder(sequence)
        
        return torch.tensor(one_hot_sequence, dtype=torch.float), label


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
   
    train_sequences = read_sequences('train.txt')
    test_sequences = read_sequences('test.txt')

   
    train_dataset = CustomDataset(train_sequences)
    test_dataset = CustomDataset(test_sequences)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    model = AlexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in range(1, 10 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()


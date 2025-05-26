import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f"Label: {example_targets[i].item()}")
    plt.axis('off')
plt.show()

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return running_loss/len(train_loader), correct/total

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return test_loss/len(test_loader), correct/total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(epochs):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
          f"Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%")
    
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1,2,2)
plt.plot([acc for acc in train_accs], label='Train Acc')
plt.plot([acc for acc in test_accs], label='Test Acc')
plt.legend()
plt.title('Accuracy Curve')

plt.show()

#q2
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP ekledik
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class LargeKernelCNN(nn.Module):
    def __init__(self):
        super(LargeKernelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)    # 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*4*4, 128)   # Boyutlara dikkat et
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*4*4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class LeakyReLUCNN(nn.Module):
    def __init__(self):
        super(LeakyReLUCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return running_loss/len(train_loader), correct/total

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return test_loss/len(test_loader), correct/total

models = {
    "Deeper CNN": DeeperCNN,
    "Large Kernel CNN": LargeKernelCNN,
    "LeakyReLU CNN": LeakyReLUCNN
}
results = {}

for name, ModelClass in models.items():
    print(f"Training {name}...")
    model = ModelClass().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
          f"Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%")
        
    results[name] = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs
    }

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
for name, res in results.items():
    plt.plot([acc for acc in res['test_accs']], label=f"{name}")
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,2)
for name, res in results.items():
    plt.plot(res['test_losses'], label=f"{name}")
plt.title("Test Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,3)
for name, res in results.items():
    plt.plot([acc for acc in res['train_accs']], label=name)
plt.title("Train Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,4)
for name, res in results.items():
    plt.plot(res['train_losses'], label=name)
plt.title("Train Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

#q3
learning_rates = [0.1, 0.01, 0.001]

results = {}

epochs = 10
criterion = nn.CrossEntropyLoss()

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = BaselineCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")

    results[lr] = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs
    }

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
for lr in learning_rates:
    plt.plot([acc for acc in results[lr]['test_accs']], label=f"LR={lr}")
plt.title("Test Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,2)
for lr in learning_rates:
    plt.plot([acc for acc in results[lr]['test_losses']], label=f"LR={lr}")
plt.title("Test Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,3)
for lr in learning_rates:
    plt.plot([acc for acc in results[lr]['train_accs']], label=f"LR={lr}")
plt.title("Train Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,4)
for lr in learning_rates:
    plt.plot([acc for acc in results[lr]['train_losses']], label=f"LR={lr}")
plt.title("Train Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

optimizers = {
    "Adam": lambda params: optim.Adam(params, lr=0.001),
    "SGD": lambda params: optim.SGD(params, lr=0.001, momentum=0.9)
}

epochs = 10
criterion = nn.CrossEntropyLoss()

results = {}

for opt_name, opt_func in optimizers.items():
    print(f"\nTraining with optimizer: {opt_name}")
    model = BaselineCNN().to(device)
    optimizer = opt_func(model.parameters())

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")

    results[opt_name] = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs
    }

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
for opt_name, res in results.items():
    plt.plot([acc for acc in res['test_accs']], label=opt_name)
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,2)
for opt_name, res in results.items():
    plt.plot([acc for acc in res['test_losses']], label=opt_name)
plt.title("Test Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,3)
for opt_name, res in results.items():
    plt.plot([acc for acc in res['train_accs']], label=opt_name)
plt.title("Train Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,4)
for opt_name, res in results.items():
    plt.plot([acc for acc in res['train_losses']], label=opt_name)
plt.title("Train Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

batch_sizes = [32, 64, 128]

results = {}
epochs = 10
criterion = nn.CrossEntropyLoss()

for bs in batch_sizes:
    print(f"\nTraining with batch size: {bs}")

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # test batch size sabit kalabilir

    model = BaselineCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")

    results[bs] = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs
    }

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
for bs in batch_sizes:
    plt.plot([acc for acc in results[bs]['test_accs']], label=f"Batch size {bs}")
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,2)
for bs in batch_sizes:
    plt.plot([acc for acc in results[bs]['test_losses']], label=f"Batch size {bs}")
plt.title("Test Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,3)
for bs in batch_sizes:
    plt.plot([acc for acc in results[bs]['train_accs']], label=f"Batch size {bs}")
plt.title("Train Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,4)
for bs in batch_sizes:
    plt.plot([acc for acc in results[bs]['train_losses']], label=f"Batch size {bs}")
plt.title("Train Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

class BaselineCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):  # Varsayılan 0.5
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.dropout = nn.Dropout(dropout_rate)  # Parametreyi kullan
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
dropout_rates = [0.2, 0.5, 0.8]

batch_size = 64
epochs = 10

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()

results = {}

for rate in dropout_rates:
    print(f"\nTraining with dropout rate: {rate}")
    model = BaselineCNN(dropout_rate=rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")

    results[rate] = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs
    }

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
for rate in dropout_rates:
    plt.plot([acc for acc in results[rate]['test_accs']], label=f"Dropout {rate}")
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,2)
for rate in dropout_rates:
    plt.plot([acc for acc in results[rate]['test_losses']], label=f"Dropout {rate}")
plt.title("Test Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,3)
for rate in dropout_rates:
    plt.plot([acc for acc in results[rate]['train_accs']], label=f"Dropout {rate}")
plt.title("Train Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,4)
for rate in dropout_rates:
    plt.plot([acc for acc in results[rate]['train_losses']], label=f"Dropout {rate}")
plt.title("Train Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

def init_weights_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_normal(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return running_loss / len(train_loader), correct / total

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return test_loss / len(test_loader), correct / total

init_methods = {
    "Xavier": init_weights_xavier,
    "Kaiming": init_weights_kaiming,
    "Normal": init_weights_normal
}

epochs = 10
criterion = nn.CrossEntropyLoss()

results = {}

for name, init_func in init_methods.items():
    print(f"\nTraining with weight init: {name}")
    model = BaselineCNN().to(device)
    model.apply(init_func)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")

    results[name] = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs
    }

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
for name, res in results.items():
    plt.plot([acc for acc in res['test_accs']], label=name)
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,2)
for name, res in results.items():
    plt.plot(res['test_losses'], label=name)
plt.title("Test Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,2,3)
for name, res in results.items():
    plt.plot([acc for acc in res['train_accs']], label=name)
plt.title("Train Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2,2,4)
for name, res in results.items():
    plt.plot(res['train_losses'], label=name)
plt.title("Train Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

#q4
model = BaselineCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.5)  # high lr

epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(epochs):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
    test_loss, test_acc = test(model, device, test_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title("Loss Curve (High LR)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,1,2)
plt.plot([acc for acc in train_accs], label='Train Accuracy')
plt.plot([acc for acc in test_accs], label='Test Accuracy')
plt.title("Accuracy Curve (High LR)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

class HighDropoutCNN(nn.Module):
    def __init__(self):
        super(HighDropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.dropout = nn.Dropout(0.95)  # high dropout
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return running_loss / len(train_loader), correct / total

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return test_loss / len(test_loader), correct / total

model = HighDropoutCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(epochs):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
    test_loss, test_acc = test(model, device, test_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
          f"Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%")
    
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title("Loss Curve (High Dropout)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,1,2)
plt.plot([acc for acc in train_accs], label='Train Accuracy')
plt.plot([acc for acc in test_accs], label='Test Accuracy')
plt.title("Accuracy Curve (High Dropout)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1   = nn.Linear(64*5*5, 128)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _,pred = out.max(1)
        correct  += pred.eq(y).sum().item()
        total    += y.size(0)
    return total_loss/len(loader), correct/total

def test_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            out  = model(X)
            loss = criterion(out,y)
            total_loss += loss.item()
            _,pred = out.max(1)
            correct  += pred.eq(y).sum().item()
            total    += y.size(0)
    return total_loss/len(loader), correct/total


model     = BaselineCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,      
                      momentum=0.99 # high sgd
                     )

epochs = 10
train_losses, train_accs = [], []
test_losses,  test_accs  = [], []

for ep in range(1, epochs+1):
    tl, ta = train_epoch(model, train_loader, optimizer, criterion)
    vl, va = test_epoch (model, test_loader,  criterion)
    train_losses.append(tl); train_accs.append(ta)
    test_losses .append(vl); test_accs .append(va)
    print(f"Epoch {ep}: Train Acc={ta*100:.2f}% | Test Acc={va*100:.2f}%")

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses,  label='Test Loss')
plt.title("Loss Curve (SGD, momentum=0.99)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(2,1,2)
plt.plot([a*100 for a in train_accs], label='Train Acc')
plt.plot([a*100 for a in test_accs],  label='Test Acc')
plt.title("Accuracy Curve (SGD, momentum=0.99)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()

plt.tight_layout()
plt.show()
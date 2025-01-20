# Package 준비
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 

from torchvision import datasets
from torchvision import transforms

# MNIST dataset
mnist_train = datasets.MNIST(root='./data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
print("Downloading Train Data Done ! ")

mnist_test = datasets.MNIST(root='./data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
print("Downloading Test Data Done ! ")

batch_size = 256

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# Defining Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = X.reshape(X.size(0), -1)
        X = self.fc(X)
        return X

model = Model()

# 학습 방법 준비
## 오차함수 
## Greaidnet descent 알고리즘

LEARNING_RATE = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training
epochs = 5

for epoch in range(epochs):
    model.train()
    avg_loss = 0
    avg_acc = 0
    total = 0
    correct = 0
    for i, (batch_img, batch_lab) in enumerate(train_loader):
        y_pred = model.forward(batch_img)

        loss = criterion(y_pred, batch_lab)

        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        _, predicted = torch.max(y_pred.data, 1)
        total += batch_lab.size(0)
        correct += (predicted == batch_lab).sum().item()
        
    acc = (100 * correct / total)

    model.eval()
    with torch.no_grad():
        val_loss = 0
        total = 0
        correct = 0
        for i, (batch_img, batch_lab) in enumerate(val_loader):
            y_pred = model(batch_img)
            val_loss += criterion(y_pred, batch_lab)
            _, predicted = torch.max(y_pred.data, 1)
            total += batch_lab.size(0)
            correct += (predicted == batch_lab).sum().item()
            
        val_loss /= len(val_loader)
        val_acc = (100 * correct / total)
        
    print(f"Epoch : {epoch+1}, Loss : {(avg_loss/len(train_loader)):.3f}, Acc: {acc:.3f}, Val Loss : {val_loss.item():.3f}, Val Acc : {val_acc:.3f}\n")

print("Training Done !")
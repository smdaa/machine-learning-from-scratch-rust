#!/usr/bin/env python3

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=10)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# load data
x_train = np.loadtxt("x_train.txt")
x_test = np.loadtxt("x_test.txt")
y_train = np.loadtxt("y_train.txt")
y_test = np.loadtxt("y_test.txt")

print("x_train.shape :", x_train.shape)
print("x_test.shape :", x_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape :", y_test.shape)

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()

y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

train_data = torch.utils.data.TensorDataset(x_train, y_train)
test_data = torch.utils.data.TensorDataset(x_test, y_test)

# load model
model = Net()
learning_rate = 0.01
momentum = 0.0
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
loss_fn = torch.nn.CrossEntropyLoss()

# train model
epochs = 20
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=False
)

total_step = len(train_loader)

start = time.time()
for epoch in range(epochs):
    running_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        _, truth = labels.max(1)
        total += labels.size(0)
        correct += predicted.eq(truth).sum().item()

    train_loss = running_loss / total_step
    accuracy = correct / total

    print(
        "Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}, Accuracy: {:.3f}".format(
            epoch + 1, epochs, i + 1, total_step, train_loss, accuracy
        )
    )


# test model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        _, truth = labels.max(1)
        total += labels.size(0)
        correct += predicted.eq(truth).sum().item()

    accuracy = correct / total
    print("Test accuracy : %.3f" % accuracy)

end = time.time()
print("elapsed time", end - start)

from Model import Model
from conv2d import Conv2D
from linear import Linear
from relu import Relu
from MaxPool2d import MaxPool2d
from flatten import Flatten
from AdamW import AdamW
from CosineAnnealingLR import CosineAnnealingLR
from CrossEntropyLoss import CrossEntropyLoss
from Dropout import Dropout
import torchvision
from torchvision.datasets import CIFAR10
import numpy as np

def augment_image(x, flip_prob=0.6, pad=4):
    if np.random.rand() > flip_prob:
        x = np.flip(x, axis=-1).copy()

    if np.random.rand() > 0.8 and pad > 0:
        x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
        H, W = x.shape[1], x.shape[2]
        h_start = np.random.randint(0, 2 * pad + 1)
        w_start = np.random.randint(0, 2 * pad + 1)
        x = x_padded[:, h_start:h_start + H, w_start:w_start + W].copy()

    return x

# train_dataset = torchvision.datasets.MNIST(
#     root="./MNIST/train", train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=True)
#
# test_dataset = torchvision.datasets.MNIST(
#     root="./MNIST/test", train=False,
#     transform=torchvision.transforms.ToTensor(),
#     download=True)

cifar = CIFAR10(root="./data", train=True, download=True)
X_small = np.array([
    np.transpose(np.array(img).astype(np.float32) / 255.0, (2,0,1))
    for img, _ in cifar
])
mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
X_small = (X_small - mean) / std
y_small = np.array([label for _, label in cifar])

cifar_test = CIFAR10(root="./data", train=False, download=True)
X_test = np.array([
    np.transpose(np.array(img).astype(np.float32) / 255.0, (2,0,1))
    for img, _ in cifar_test
])
X_test = (X_test - mean) / std
y_test = np.array([label for _, label in cifar_test])

def split_dataset(X, y, val_ratio=0.1, random_state=42):
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    val_size = int(n_samples * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])

# X_train_full = np.array([img.numpy().squeeze() for img, _ in train_dataset])
# y_train_full = np.array([label for _, label in train_dataset])
# X_test = np.array([img.numpy().squeeze() for img, _ in test_dataset])
# y_test = np.array([label for _, label in test_dataset])
#
# X_train_full = X_train_full[:, np.newaxis, :, :]
# X_test = X_test[:, np.newaxis, :, :]

(X_train, y_train), (X_val, y_val) = split_dataset(X_small, y_small, val_ratio=0.2)
# (X_train, y_train), (X_val, y_val) = split_dataset(X_train_full, y_train_full, val_ratio=0.15)

class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=False):
        self.X = np.array(X)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            X_batch = self.X[batch_idx]
            if self.shuffle:
                X_batch = np.array([augment_image(img) for img in X_batch])
            yield X_batch, self.y[batch_idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


model = Model()
# model.add(Conv2D(1, 8,(3,3)))
# model.add(Relu())
# model.add(MaxPool2d((2,2)))
# model.add(Flatten())
# model.add(Linear(13*13*8,10))
#
model.add(Conv2D(3, 32, (3,3), padding=1))
model.add(Relu())
model.add(Conv2D(32, 32, (3,3), padding=1))
model.add(Relu())
model.add(MaxPool2d((2,2)))

model.add(Conv2D(32, 64, (3,3), padding=1))
model.add(Relu())
model.add(MaxPool2d((2,2)))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Linear(4096, 128))
model.add(Relu())
model.add(Dropout(0.3))
model.add(Linear(128, 10))

train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=64, shuffle=False)
test_loader = DataLoader(X_test, y_test, batch_size=64)

EPOCHS = 15
optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))
criterion = CrossEntropyLoss()

arr_of_train_loss = []
arr_of_val_loss = []
for epoch in range(EPOCHS):
    train_loss = 0
    model.train()
    for x, labels in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        train_loss += loss
        dout = criterion.backward()
        model.backward(dout)
        optimizer.step()
        scheduler.step()
    train_loss/=len(train_loader)
    arr_of_train_loss.append(train_loss)

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    for x, labels in val_loader:
        logits = model(x)
        loss = criterion(logits, labels)
        val_loss+=loss

        preds = np.argmax(logits, axis=1)
        val_correct += np.sum(preds == labels)
        val_total += len(labels)
    val_loss /= len(val_loader)
    arr_of_val_loss.append(val_loss)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch + 1}/{EPOCHS} | " f"Train Loss: {train_loss:.4f} | " f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

model.eval()
test_correct = 0
test_total = 0
for x, labels in test_loader:
    logits = model(x)

    preds = np.argmax(logits, axis=1)
    test_correct += np.sum(preds == labels)
    test_total += len(labels)
test_acc = test_correct / test_total

print(f"Test Acc: {test_acc:.4f}")


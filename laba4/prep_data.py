import numpy as np
import matplotlib.pyplot as plt

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

def split_dataset(X, y, val_ratio=0.1, random_state=42):
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    val_size = int(n_samples * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])

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

def loss_graph(epochs, train_loss, val_loss):
    plt.figure(figsize=(10,5))

    plt.plot(epochs, train_loss, color="green", label="train loss")
    plt.plot(epochs, val_loss, color="red", label="val loss")

    plt.xlabel("эпоха")
    plt.ylabel("loss")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def metric_graph(epochs, accuracy):
    plt.figure(figsize=(10,5))

    plt.plot(epochs, accuracy, color="blue", label="accuracy")

    plt.xlabel("эпоха")
    plt.ylabel("значениe")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


correct_examples = []
incorrect_examples = []
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
STD = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)


def denormalize(img):
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def plot_example(ax, img, true_l, pred_l, logit, is_correct):
    ax.imshow(np.transpose(img, (1, 2, 0)))
    ax.axis('off')

    max_l = np.max(logit)
    exp_logit = np.exp(logit - max_l)
    conf = exp_logit[pred_l] / np.sum(exp_logit)

    color = 'green' if is_correct else 'red'
    status = "Correct" if is_correct else "Incorrect"

    title = (f"{status} True: {cifar_classes[true_l]}\n"
             f"Pred: {cifar_classes[pred_l]} ({conf:.1%})")

    ax.set_title(title, color=color, fontsize=9, pad=10)

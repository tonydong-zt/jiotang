import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# 1. 基本配置
# =========================

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前使用设备:", device)


# =========================
# 2. 定义 MLP 模型
# =========================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

        self.init_weights()

    def init_weights(self):
        """
        参数初始化：
        Linear 层的 weight 使用 Xavier 初始化
        Linear 层的 bias 初始化为 0
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  #从一个均匀分布中随机取数来初始化权重
                nn.init.zeros_(layer.bias)  #偏置初始化为0

    def forward(self, x):
        return self.net(x)


# =========================
# 3. 生成数据集
# =========================

def create_dataset(noise=0.2, n_samples=1000, batch_size=32):
    """
    make_moons 会生成两个月牙形交错分布的数据，用于二分类。
    noise 越大，点越分散，分类越困难。
    """

    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=RANDOM_SEED
    )

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # 归一化 / 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转成 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return X_train, X_test, y_train, y_test, train_loader, test_loader, scaler


# =========================
# 4. 可视化原始数据
# =========================

def plot_dataset(X_train, X_test, y_train, y_test, noise):
    plt.figure(figsize=(6, 5))

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        s=20,
        alpha=0.8,
        label="train"
    )

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        s=20,
        marker="x",
        alpha=0.8,
        label="test"
    )

    plt.title(f"make_moons dataset, noise={noise}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================
# 5. 计算准确率
# =========================

def evaluate(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


# =========================
# 6. 训练模型
# =========================

def train_model(noise=0.2, batch_size=32, epochs=200, lr=0.01, show_plot=True):
    X_train, X_test, y_train, y_test, train_loader, test_loader, scaler = create_dataset(
        noise=noise,
        batch_size=batch_size
    )

    model = MLP().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accs = []
    test_accs = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            logits = model(X_batch)

            # 损失计算
            loss = loss_fn(logits, y_batch)

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        avg_loss = total_loss / total_samples
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % 50 == 0:
            print(
                f"noise={noise}, batch_size={batch_size}, "
                f"epoch={epoch + 1}, "
                f"loss={avg_loss:.4f}, "
                f"train_acc={train_acc:.4f}, "
                f"test_acc={test_acc:.4f}"
            )

    train_time = time.time() - start_time

    if show_plot:
        plot_training_curves(train_losses, train_accs, test_accs, noise, batch_size)
        plot_decision_boundary(model, X_train, X_test, y_train, y_test, scaler, noise)

    return model, train_time, test_accs[-1], scaler


# =========================
# 7. 可视化 loss 和 accuracy
# =========================

def plot_training_curves(train_losses, train_accs, test_accs, noise, batch_size):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_losses)
    plt.title(f"Training Loss, noise={noise}, batch_size={batch_size}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, test_accs, label="Test Accuracy")
    plt.title(f"Accuracy, noise={noise}, batch_size={batch_size}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================
# 8. 可视化决策边界 / 热力图
# =========================

def plot_decision_boundary(model, X_train, X_test, y_train, y_test, scaler, noise):
    model.eval()

    X_all = np.vstack([X_train, X_test])

    x_min, x_max = X_all[:, 0].min() - 0.8, X_all[:, 0].max() + 0.8
    y_min, y_max = X_all[:, 1].min() - 0.8, X_all[:, 1].max() + 0.8

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    start_time = time.time()

    with torch.no_grad():
        logits = model(grid_tensor)
        probs = torch.softmax(logits, dim=1)
        class1_prob = probs[:, 1].cpu().numpy()

    infer_time = time.time() - start_time

    zz = class1_prob.reshape(xx.shape)

    plt.figure(figsize=(6, 5))

    plt.contourf(xx, yy, zz, levels=50, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Probability of class 1")

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        s=20,
        edgecolors="k",
        alpha=0.8,
        label="train"
    )

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        s=20,
        marker="x",
        alpha=0.9,
        label="test"
    )

    plt.title(f"Decision Boundary Heatmap, noise={noise}\nInference time: {infer_time:.4f}s")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"热力图推理耗时: {infer_time:.4f} 秒")


# =========================
# 9. 对比不同 noise 的效果
# =========================

def experiment_noise():
    noise_list = [0.05, 0.2, 0.4]

    results = []

    for noise in noise_list:
        print("\n==============================")
        print(f"开始训练 noise = {noise}")
        print("==============================")

        model, train_time, test_acc, scaler = train_model(
            noise=noise,
            batch_size=32,
            epochs=200,
            lr=0.01,
            show_plot=True
        )

        results.append((noise, train_time, test_acc))

    print("\n不同 noise 的结果对比：")
    print("noise\ttraining_time(s)\ttest_accuracy")

    for noise, train_time, test_acc in results:
        print(f"{noise}\t{train_time:.4f}\t\t{test_acc:.4f}")


# =========================
# 10. 对比三种 batch size 的速度
# =========================

def experiment_speed():
    batch_sizes = [1, 32, 700]

    results = []

    for batch_size in batch_sizes:
        print("\n==============================")
        print(f"开始训练 batch_size = {batch_size}")
        print("==============================")

        model, train_time, test_acc, scaler = train_model(
            noise=0.2,
            batch_size=batch_size,
            epochs=200,
            lr=0.01,
            show_plot=False
        )

        results.append((batch_size, train_time, test_acc))

    print("\n三种 batch size 的训练速度对比：")
    print("batch_size\ttraining_time(s)\ttest_accuracy")

    for batch_size, train_time, test_acc in results:
        print(f"{batch_size}\t\t{train_time:.4f}\t\t{test_acc:.4f}")


# =========================
# 主程序
# =========================

if __name__ == "__main__":
    # 先画一个数据集看看
    X_train, X_test, y_train, y_test, _, _, _ = create_dataset(
        noise=0.2,
        batch_size=32
    )
    plot_dataset(X_train, X_test, y_train, y_test, noise=0.2)

    # 实验一：对比不同 noise
    experiment_noise()

    # 实验二：对比三种 batch size 的速度
    experiment_speed()
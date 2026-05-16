
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. 加载加州房价数据集
data=fetch_california_housing()

X= data.data
y=data.target

print("特征X的形状：",X.shape)
print("目标值y的形状：",y.shape)
print("特征名称：",data.feature_names)


#2.拆分训练集和测试集
X_train, X_test, y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("训练集 X_train的形状：",X_train.shape)
print("测试集 X_test的形状:",X_test.shape)


#3. 标准化特征
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#4.转成Tensor
X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)

y_train=torch.tensor(y_train,dtype=torch.float32).view(-1,1) #-1代表torch自动推断，以转换成特定shape，去接受linaer的输出
y_test=torch.tensor(y_test,dtype=torch.float32).view(-1,1)

#5. Dataset 和 Dataloader
train_dataset=TensorDataset(X_train,y_train) #把feature和label捆绑，形成torch可使用的数据集对象
test_dataset=TensorDataset(X_test,y_test)

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

#6. define the model
class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.linear=nn.Linear(input_dim,1)#线性回归相当于没有隐藏层的神经网络
    def forward(self,x):
        return self.linear(x)

model=LinearRegressionModel(input_dim=8)


#7. 损失函数和优化器
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#8.训练
epochs=100

for epoch in range(epochs):
    model.train()
    total_loss=0

    for X_batch,y_batch in train_loader:
        pred=model(X_batch)
        loss=loss_fn(pred,y_batch)
        total_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(epoch+1)%10==0:
            print(f"epoch{epoch+1},loss={total_loss/len(train_loader):.6f}")


#9. 测试
model.eval()

with torch.no_grad():
    y_pred = model(X_test)

    # 计算 MSE
    mse = torch.mean((y_pred - y_test) ** 2)

    # 计算 R²
    ss_res = torch.sum((y_test - y_pred) ** 2)
    ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("测试集 MSE：", mse.item())
    print("测试集 R²：", r2.item())

    # 绘制散点图
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("真实房价")
    plt.ylabel("预测房价")
    plt.title("真实房价 vs 预测房价")

    # 如果预测完美，所有点应该落在 y = x 这条直线上
    min_value = torch.min(torch.min(y_test), torch.min(y_pred))
    max_value = torch.max(torch.max(y_test), torch.max(y_pred))

    plt.plot(
        [min_value.item(), max_value.item()],
        [min_value.item(), max_value.item()]
    )

    plt.show()

    
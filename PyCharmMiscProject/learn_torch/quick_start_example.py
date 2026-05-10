import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

#------------------
# Loading data in pytorch
#------------------

# Download training data from open datasets.
training_data=datasets.FashionMNIST(
    root="data",    #the download address of the dataset
    train=True,     #a training dataset
    download=True,  # if don't have this dataset,download it automatically
    transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]),
)

# compose indeicates exacute following things step by step
#ToInmage: turning original picturn into image/tensor form,making it easier to be precessed
# turn int into float,and turn pixel value into float in 0~1;


# Download test data from open datasets.
test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]),
)

#test the shape of data:
# image,label=training_data[0]
# print(image.shape)
# print(label)

#use dataloader to batch data
batch_size=64

# create data loaders
train_dataloader=DataLoader(training_data,batch_size=batch_size)
test_dataloader=DataLoader(test_data,batch_size=batch_size)

for X,y in test_dataloader:
    print(f"Shaper of X [N,C,H,W]:{X.shape}")
    print(f"Shape of y:{y.shape} {y.dtype}")

# N = batch size，也就是样本数量
# C = channel，通道数
# H = height，高度
# W = width，宽度

# y 里面有 64 个标签
# 每个标签是一个整数类别编号


#-----------------
# Creating models
#-----------------

device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.liner_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x=self.flatten(x)
        logits=self.liner_relu_stack(x)
        return logits
model=NeuralNetwork().to(device)
print(model)


#---------------------------------
# Optimizing the Model Parameters
#---------------------------------

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)

def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        # compute prediction error
        pred=model(X)
        loss=loss_fn(pred,y)

        #BackPropagation
        loss.backward() # 反向传播，计算参数
        optimizer.step() #根据计算出的梯度，修改模型参数
        optimizer.zero_grad()# 清空梯度，防止梯度累加

        if batch%100 == 0:
            loss,current=loss.item(),(batch+1)*len(X) #item将tensor类型转为python数字类型的loss
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset) #数据集大小
    num_batches=len(dataloader)  #batch数量
    model.eval()  #转换为测试模式
    test_loss,correct=0,0  #累计测试损失，累计预测正确的样本数
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item() #sum:把batch中的所有值加起来
    test_loss /= num_batches
    correct /= size
    print(f"Test Error :\n Acurracy:{(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#epochs 表示把训练集的所有数据全部过一遍的轮数

epochs=5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader,model ,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("Done!")



#------------------
# saving models
#------------------

torch.save(model.state_dict(),"model.pth")#取出模型参数字典 保存到本地的文件名
print("Saved PyTorch Model State to model.pth")




#-----------------
# Loading models
#-----------------

model=NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth",weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() #测试模式下不改变参数
x,y=test_data[0][0],test_data[0][1]
with torch.no_grad():  #关闭梯度计算，加速，减内存，减计算
    x=x.to(device)
    pred=model(x)
    predicted,actual=classes[pred[0].argmax(0)],classes[y]  #取出pred[0]之后已经是一维向量，此时argmax（0）就说得通
    print(f'predicted:"{predicted}",Actual: "{actual}"')
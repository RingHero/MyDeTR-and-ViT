import torchvision as tv
import torchvision.transforms as transforms
import torch
from torchvision.transforms import ToPILImage
import vision_transformer
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
show = ToPILImage() # 可以把Tensor转成Image，方便可视化

# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约160M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化 先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1) 
                             ])                    

# 训练集（因为torchvision中已经封装好了一些常用的数据集，包括CIFAR10、MNIST等，所以此处可以这么写 tv.datasets.CIFAR10()）
trainset = tv.datasets.CIFAR10(
                    root='./data/',   # 将下载的数据集压缩包解压到当前目录的DataSet目录下
                    train=True, 
                    download=False,    # 如果之前没手动下载数据集，这里要改为True
                    transform=transform)

trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=100,
                    shuffle=True, 
                    num_workers=0)

# 测试集
testset = tv.datasets.CIFAR10(
                    './data/',
                    train=False, 
                    download=False,   # 如果之前没手动下载数据集，这里要改为True 
                    transform=transform)

testloader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=100, 
                    shuffle=False,
                    num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vision_transformer.MyViT(n_dim=32,cls_num=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoches = 50
    hst_loss = []
    for epoch in trange(epoches):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):   # i 第几个batch     data：一个batch中的数据
            
            # 输入数据
            inputs, labels = data   # images：batch大小为4     labels：batch大小为4
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(inputs.shape)
            # 梯度清零
            optimizer.zero_grad()
            
            # forward + backward 
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()   
            
            # 更新参数 
            optimizer.step()
            
            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 200 == 199: # 每2000个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' \
                    % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
        hst_loss.append(running_loss)
    print('Finished Training')
    torch.save(model.state_dict(), "vit_model.pth")
    times = np.arange(epoches)
    plt.plot(times,hst_loss)
    plt.show()


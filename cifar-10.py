import torchvision
import torchvision.transforms as transforms
 
# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
 
# 定义数据存储目录，你可以根据需要修改
data_dir = './data'
 
# 下载 CIFAR10 训练集
train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True, transform=transform)
 
# 下载 CIFAR10 测试集
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform)
 
print("CIFAR10 数据集下载完成！")
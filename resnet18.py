import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision


class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1) -> None:
        super().__init__()
        self.straight = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,1,stride,bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self,x):
        out = self.straight(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1) -> None:
        super().__init__()
        self.straight = nn.Sequential(
            nn.Conv2d(in_channel,int(out_channel/4),1,1,bias=False),
            nn.BatchNorm2d(int(out_channel/4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channel/4),int(out_channel/4),3,stride,padding=1,bias=False),
            nn.BatchNorm2d(int(out_channel/4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channel/4),out_channel,1,1,bias=False),
            nn.BatchNorm2d(out_channel*4)
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,out_channel*4,1,stride,bias=False),
                nn.BatchNorm2d(out_channel*4)
            )
    
    def forward(self,x):
        out = self.straight(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet18(nn.Module):
    def __init__(self,num_classes=1000) -> None:
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,self.in_channel,7,2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1) 
        )
        self.layer1 = self.make_layer(BasicBlock,64,64,1)
        self.layer2 = self.make_layer(BasicBlock,64,128,2)
        self.layer3 = self.make_layer(BasicBlock,128,256,2)
        self.layer4 = self.make_layer(BasicBlock,256,512,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes,bias=True)

    def make_layer(self,block,in_channel,out_channel,stride):
        layers = []
        layers.append(block(in_channel,out_channel,stride))
        layers.append(block(out_channel,out_channel,1))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = self.avgpool(out)
        #print(out.shape)
        out = out.view(out.size(0),-1)
        #print(out.shape)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    #model = torchvision.models.resnet18(pretrained=True)
    model = ResNet18()
    model.load_state_dict(torch.load("resnet18.pth"),strict=False)
    model = torch.nn.Sequential(*list(model.children())[:-2])
    print(model)

    # 计算所有参数数量
    all_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {all_params}")

    x = torch.randn(1,3,224,224)
    out = model(x)
    print(out.shape)

    
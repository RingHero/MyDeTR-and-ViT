import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self,n_dim=256):
        super().__init__()
        self.linear = nn.Linear(n_dim,n_dim)

    def forward(self,x):
        return self.linear(x)

class MixtureOfExperts(nn.Module):
    def __init__(self,n_expert=4,n_dim=256):
        super().__init__()
        self.gate = nn.Linear(n_dim,n_expert)
        self.experts = nn.ModuleList([Expert(n_dim) for _ in range(n_expert)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        gate = self.softmax(self.gate(x))
        return sum([gate[:,:,i].unsqueeze(-1)*self.experts[i](x) for i in range(len(self.experts))])
    
if __name__ == "__main__":
    model = MixtureOfExperts(n_expert=4,n_dim=256)
    x = torch.randn(1,10,256)   #[bs,token_nums,n_dim]
    print(model(x).shape)
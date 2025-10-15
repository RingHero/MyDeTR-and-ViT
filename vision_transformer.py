import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self,embed_dim=256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.Conv2d(64,128,4,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,self.embed_dim,3)
        )
        self.num_patches = 25
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim)#这里加1 是因为x后续会加入cls_token
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        #print(self.cls_token.shape)

    def forward(self,x):
        bs = x.shape[0]
        x = self.backbone(x)
        x = x.flatten(2).transpose(-2,-1)
        cls_token = self.cls_token.repeat(bs,1,1)#  cls_token = [bs,1,256]
        x = torch.cat([x,cls_token],dim=1)
        x = x + self.position_embeddings
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,n_head=8,n_dim=256):
        super().__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.head_dim = n_dim//n_head
        assert self.head_dim*n_head == n_dim

        self.w_q = nn.Linear(self.n_dim,self.n_dim)#  256->256
        self.w_k = nn.Linear(self.n_dim,self.n_dim)
        self.w_v = nn.Linear(self.n_dim,self.n_dim)
        self.w_o = nn.Linear(self.n_dim,self.n_dim)
    
    def forward(self,x,mask=None):#   x = [bs,token_nums,n_dim]
        bs, n_tokens, d_model = x.shape
        #print(x.shape)
        q = self.w_q(x)#   q = k = v = [bs,token_nums,n_dim]
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(bs,n_tokens,self.n_head,self.head_dim)#   q = k = v = [bs,token_nums,n_head,head_dim]
        k = k.view(bs,n_tokens,self.n_head,self.head_dim)
        v = v.view(bs,n_tokens,self.n_head,self.head_dim)

        q = q.permute(0,2,1,3).contiguous()#    q = k = v = [bs,n_head,token_nums,head_dim]
        k = k.permute(0,2,1,3).contiguous()
        v = v.permute(0,2,1,3).contiguous()

        scores = torch.matmul(q,k.transpose(-2,-1))/ math.sqrt(self.head_dim)#  scores = [bs,n_head,token_nums,token_nums]

        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))
        
        attn_weights = F.softmax(scores,dim=-1)
        context = torch.matmul(attn_weights,v)# context = [bs,n_head,token_nums,head_dim]
        context = context.permute(0,2,1,3).contiguous()# context = [bs,token_nums,n_head,head_dim]
        context = context.view(bs,n_tokens,self.n_dim)# context = [bs,n_tokens,n_dim]
        output = self.w_o(context)

        return output

class CrossAttention(nn.Module):
    def __init__(self,n_head=8,n_dim=256):
        super().__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.head_dim = n_dim//n_head
        assert self.head_dim*n_head == n_dim

        self.w_q = nn.Linear(self.n_dim,self.n_dim)
        self.w_k = nn.Linear(self.n_dim,self.n_dim)
        self.w_v = nn.Linear(self.n_dim,self.n_dim)
        self.w_o = nn.Linear(self.n_dim,self.n_dim)

    def forward(self,query,key,value,mask=None):
        bs,Nq,q_dim = query.shape
        bs,Nk,k_dim = key.shape
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(bs,Nq,self.n_head,self.head_dim)
        k = k.view(bs,Nk,self.n_head,self.head_dim)
        v = v.view(bs,Nk,self.n_head,self.head_dim)

        q = q.permute(0,2,1,3).contiguous()
        k = k.permute(0,2,1,3).contiguous()
        v = v.permute(0,2,1,3).contiguous()

        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(scores,float('-inf'))

        context = torch.matmul(scores,v)
        context = context.permute(0,2,1,3).contiguous().view(bs,Nq,self.n_dim)
        output = self.w_o(context)
        return output

class FeedForward(nn.Module):
    def __init__(self,n_dim=256,hidden_dim = 512,dropout_rate = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim,n_dim)
        )
    
    def forward(self,x):
        output = self.ffn(x)
        return output

class EncoderBlock(nn.Module):
    def __init__(self,n_head=8,n_dim=48,drop_out=0.1):
        super().__init__()
        self.multiheadattn = MultiHeadAttention(n_head=n_head,n_dim=n_dim)
        self.ffn = FeedForward(n_dim=n_dim)
        self.norm1 = nn.LayerNorm(n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self,x):
        attn_out = self.dropout1(self.multiheadattn(x))
        x = self.norm1(x + attn_out)
        ffn_out = self.dropout2(self.ffn(x))
        x = self.norm2(x + ffn_out)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,n_head=8,n_dim=256,drop_out=0.1):
        super().__init__()
        self.selfattn = MultiHeadAttention(n_head=n_head,n_dim=n_dim)
        self.crossattn = CrossAttention(n_head=n_head,n_dim=n_dim)
        self.ffn = FeedForward(n_dim=n_dim)
        self.norm1 = nn.LayerNorm(n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.norm3 = nn.LayerNorm(n_dim)
        self.drop_out1 = nn.Dropout(drop_out)
        self.drop_out2 = nn.Dropout(drop_out)
        self.drop_out3 = nn.Dropout(drop_out)

    def forward(self,query,key,value):
        x = self.norm1(query + self.drop_out1(self.selfattn(query)))
        x = self.norm2(x+self.drop_out2(self.crossattn(x,key,value)))
        x = self.norm3(self.drop_out3(self.ffn(x)))
        return x

class Encoder(nn.Module):
    def __init__(self,num_layers=6,n_dim=256):
        super().__init__()
        self.n_dim = n_dim
        self.layers = nn.ModuleList([EncoderBlock(n_dim=n_dim) for _ in range(num_layers)])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(num_layers)])
    
    def forward(self,query,key,value):
        for layer in self.layers:
            query = layer(query,key,value)
        return query
    
class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x):
        memory = self.encoder(x)
        return self.decoder(x,memory,memory)

class MyViT(nn.Module):
    def __init__(self,n_dim=256,cls_num=10) -> None:
        super().__init__()
        self.encoder = Encoder(num_layers=4,n_dim=n_dim)
        self.patch_emb = PatchEmbedding(embed_dim=n_dim)
        self.mlp_classifier = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Linear(n_dim,cls_num)
        )
    
    def forward(self,x):
        x = self.patch_emb(x)
        x = self.encoder(x)
        x = x[:,0]
        x = self.mlp_classifier(x)
        return x

if __name__ == "__main__":
    model = MyViT(n_dim=48,cls_num=10)
    input = torch.rand(4,3,32,32)    #  [bs,n_tokens,n_dim]
    output = model(input)
    #print(output.shape)
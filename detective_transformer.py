import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet18 import ResNet18
import math

class Backbone(nn.Module):
    def __init__(self,embed_dim=256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(512,embed_dim,kernel_size=1,stride=1,padding=0)
        self.backbone = nn.Sequential(*list(ResNet18().children())[:-2])
        self.backbone.load_state_dict(torch.load("resnet18.pth"),strict=False)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self,x):
        x = self.backbone(x)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        return x
        

class PatchEmbedding(nn.Module):
    def __init__(self,embed_dim=256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = Backbone(embed_dim)
        self.num_patches = 49
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches+1, self.embed_dim)#这里加1 是因为x后续会加入cls_token 对于VIT是这样，如果是DETR那就不加
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #print(self.cls_token.shape)

    def forward(self,x):
        bs = x.shape[0]
        #print(x.shape)
        x = self.backbone(x)
        #print("这里")
        x = x.flatten(2).transpose(-2,-1)
        #print(x.shape)
        cls_token = self.cls_token.repeat(bs,1,1)#  cls_token = [bs,1,256]
        x = torch.cat([x,cls_token],dim=1)
        #print(x.shape)
        position_embeddings = self.position_embeddings.repeat(bs,1,1)
        #print(position_embeddings.shape)
        x = x + position_embeddings
        return x
    
class QueryEmbedding(nn.Module):
    def __init__(self,n_query=100,embed_dim=256) -> None:
        super().__init__()
        self.queries = torch.zeros(1,n_query,embed_dim)
        self.query_pos = nn.Embedding(n_query,embed_dim)

    def forward(self,x):
        bs = x.shape[0]
        obj_query = self.queries.expand(bs,-1,-1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(bs,-1,-1)
        output = obj_query + query_pos
        return output


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
        bs, n_tokens, _ = x.shape
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
    def __init__(self,num_layers=6,n_dim=256):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(n_dim=n_dim) for _ in range(num_layers)])
    
    def forward(self,query,key,value):
        intermediate = []
        for layer in self.layers:
            query = layer(query,key,value)
            intermediate.append(query)
        return torch.stack(intermediate)
    
class MLP(nn.Module):
    def __init__(self,num_layers=3,input_dim=256,hidden_dim=512,output_dim=4) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim,
                                    hidden_dim if i < num_layers - 1 else output_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        output = self.layers(x)
        return output

class MyTransformer(nn.Module):
    def __init__(self,n_dim=256,cls_num=80,return_all_layer=True):
        super().__init__()
        self.encoder_layers = 6
        self.decoder_layers = 6
        self.return_all_layer = return_all_layer
        self.encoder = Encoder(num_layers=self.encoder_layers,n_dim=n_dim)
        self.decoder = Decoder(num_layers=self.decoder_layers,n_dim=n_dim)

        self.patch_emb = PatchEmbedding(embed_dim=n_dim)
        self.query_emb = QueryEmbedding(n_query=100,embed_dim=n_dim)

        self.class_head = nn.ModuleList([
            nn.Linear(n_dim,cls_num+1) for _ in range(self.decoder_layers)
        ])
        self.bbox_head = nn.ModuleList([
            MLP(num_layers=3,input_dim=n_dim,hidden_dim=512,output_dim=4) for _ in range(self.decoder_layers)
        ])

    def forward(self,x):
        x = self.patch_emb(x)
        memory = self.encoder(x)
        query = self.query_emb(x)
        decoder_outputs = self.decoder(query,memory,memory)

        all_class_logits = []
        all_bbox_preds = []
        for i in range(decoder_outputs.shape[0]):
            out = decoder_outputs[i]
            class_logit = self.class_head[i](out)
            bbox_pred = torch.sigmoid(self.bbox_head[i](out))
            all_class_logits.append(class_logit)
            all_bbox_preds.append(bbox_pred)
        
        if self.return_all_layer == True:
            return all_class_logits,all_bbox_preds
        else:
            return all_class_logits[-1],all_bbox_preds[-1]

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
    
    model = MyTransformer(n_dim=256,cls_num=80,return_all_layer=True)
    #model = Backbone()
    all_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {all_params}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {total_params}")

    encoder_input = torch.rand(2,3,224,224)    #  [bs,n_tokens,n_dim] #原图像[640,480]
    decoder_input = torch.rand(4,25,64) #[bs,,n_dim]
    cls,bbox = model(encoder_input)
    #print("cls:",cls)
    print("----------------------------------------------------------------------")
    print(cls[0].shape)
    print("----------------------------------------------------------------------")
    #print("bbox",bbox)
    print("----------------------------------------------------------------------")
    print(bbox[0].shape)
    
    


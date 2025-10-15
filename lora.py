import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    修正的 LoRA 低秩适配层
    正确实现 LoRA，冻结原始权重，只训练低秩适配矩阵
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        """
        参数:
            original_layer: 原始线性层 (nn.Linear)
            rank: LoRA 的秩（低秩矩阵的维度）
            alpha: 缩放因子
        """
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 获取原始层的参数
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # 冻结原始权重和偏置
        self.weight = nn.Parameter(original_layer.weight.data.clone(), requires_grad=False)
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # LoRA 适配矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.Tensor(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.Tensor(self.out_features, rank))
        
        # 初始化 LoRA 矩阵
        self.reset_parameters()
        
    def reset_parameters(self):
        # 初始化 LoRA 矩阵 A 和 B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # 原始权重计算
        original_output = F.linear(x, self.weight, self.bias)
        
        # LoRA 适配计算: (x * A^T) * B^T
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        
        # 合并结果并缩放
        return original_output + (self.alpha / self.rank) * lora_output

class LoRASelfAttention(nn.Module):
    """
    修正的带有 LoRA 的自注意力机制
    """
    def __init__(self, embed_size, heads, rank=8, alpha=16):
        """
        参数:
            embed_size: 嵌入维度
            heads: 注意力头的数量
            rank: LoRA 的秩
            alpha: LoRA 的缩放因子
        """
        super(LoRASelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "嵌入大小需要能被头数整除"
        
        # 创建原始线性层
        self.query_original = nn.Linear(embed_size, embed_size)
        self.key_original = nn.Linear(embed_size, embed_size)
        self.value_original = nn.Linear(embed_size, embed_size)
        self.fc_out_original = nn.Linear(embed_size, embed_size)
        
        # 使用 LoRALayer 包装原始线性层
        self.query = LoRALayer(self.query_original, rank, alpha)
        self.key = LoRALayer(self.key_original, rank, alpha)
        self.value = LoRALayer(self.value_original, rank, alpha)
        self.fc_out = LoRALayer(self.fc_out_original, rank, alpha)
        
    def forward(self, x, mask=None):
        # 获取批量大小和序列长度
        N, seq_length, _ = x.shape
        
        # 通过 LoRA 层计算 Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 重塑为多头注意力形式
        Q = Q.reshape(N, seq_length, self.heads, self.head_dim).transpose(1, 2)
        K = K.reshape(N, seq_length, self.heads, self.head_dim).transpose(1, 2)
        V = V.reshape(N, seq_length, self.heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（如果提供）
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # 计算注意力权重
        attention = F.softmax(energy, dim=-1)
        
        # 应用注意力权重到值上
        out = torch.matmul(attention, V)
        
        # 重塑回原始形状
        out = out.transpose(1, 2).reshape(N, seq_length, self.embed_size)
        
        # 通过输出层
        out = self.fc_out(out)
        
        return out

class TransformerBlockWithLoRA(nn.Module):
    """
    修正的带有 LoRA 的 Transformer 块
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion, rank=8, alpha=16):
        super(TransformerBlockWithLoRA, self).__init__()
        self.attention = LoRASelfAttention(embed_size, heads, rank, alpha)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # 前馈网络（也可以应用 LoRA，这里使用标准实现）
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力和残差连接
        attention = self.attention(x, mask)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        
        # 前馈网络和残差连接
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        x = self.dropout(x)
        
        return x

# 示例用法
if __name__ == "__main__":
    # 设置超参数
    embed_size = 256
    heads = 8
    num_layers = 6
    forward_expansion = 4
    dropout = 0.1
    rank = 8  # LoRA 秩
    alpha = 16  # LoRA 缩放因子
    
    # 创建模型
    model = nn.Sequential(
        *[TransformerBlockWithLoRA(embed_size, heads, dropout, forward_expansion, rank, alpha) 
          for _ in range(num_layers)]
    )
    
    # 创建示例输入
    batch_size = 32
    seq_length = 20
    x = torch.randn(batch_size, seq_length, embed_size)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算可训练参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {total_params}")
    
    # 计算所有参数数量
    all_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {all_params}")
    
    # 验证原始权重是否被冻结
    print("原始权重是否被冻结:")
    for name, param in model.named_parameters():
        if 'original' in name:
            print(f"{name}: {not param.requires_grad}")
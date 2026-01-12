import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================================================================
# 1. Fundamental modules: time embedding, downsampling, and upsampling
# ======================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Classic sinusoidal positional encoding, used to convert the timestep t into a vector.
    if dim = 2 * half_dim
    PE(t, 2i) = sin(t / 10e4^(i / half_dim -1))
    PE(t, 2i + 1) = cos(t / 10e4^(i / half_dim - 1))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class Downsample(nn.Module):
    """ Using stride-2 convolution for downsampling """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    """ Using transpose convolution for upsampling"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

# ======================================================================
# 2. Core modules: Attention Block
# ======================================================================

class AttentionBlock(nn.Module):
    """
    The self-attention module, 
    which is usually applied to low-resolution feature maps (e.g., 16x16, 8x8) to give the model a global receptive field.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.scale = dim ** -0.5 # sqrt(d_k)
        self.num_heads = num_heads # multi-head attention divides the channels into num_heads parts, and every part computes attention independently
        self.head_dim = dim // num_heads # every head has head_dim channels

        ## DDPM use GrounpNorm instead of BatchNorm
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim)

        ## Query, Key, Value projections for all heads
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=False)
        self.proj = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Normalization and reshape (B, C, H, W) -> (B, H*W, C)

        x_in = x
        x = self.norm(x)
        # x.permute (B, C, H, W) -> (B, H, W, C) 
        # why? Because the linear layer of transformer is applied on the last dimension (channel dimension)
        # then reshape to (B, H*W, C) it aims to change the spatial dimensions to sequence length
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 2. Generate Q, K, V

        # (B, H*W, C) -> (B, H*W, 3C) -> (B, H*W, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim) 
        # (B, H*W, 3, num_heads, head_dim) -> (3, B, num_heads, H*W, head_dim)
        # get the query, key, value tensors from the first dimension
        # Dim(q, k , v) = (B, num_heads, H*W, head_dim)
        q, k ,v = qkv.permute(2, 0, 3, 1, 4)

        # 3. Compute attention
        # scores = QK^T / sqrt(d_k)
        # Dim(scores) = (B, num_heads, H*W, H*W)
        # k.transpose(-2, -1) means that we swap the last two dimensions of k, so that we can perform the matrix multiplication between q and k^T
        # Dim(k) = (B, num_heads, H*W, head_dim)
        # Dim(k.transpose(-2, -1)) = (B, num_heads, head_dim, H*W)
        # after matmul, we get (B, num_heads, H*W, H*W)
        atten = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        atten = atten.softmax(dim=-1)

        x = torch.matmul(atten, v)  # (B, num_heads, H*W, head_dim)
        # x.transpose(1, 2) means that we swap the heads dimension and the sequence length dimension
        # reshape(B, H*W, C) means that we combine the head_dim and num_heads dimensions back to C
        x = x.transpose(1, 2).reshape(B, H * W, C)  # (B, H*W, C)

        # 4. Output projection and reshape back to (B, C, H, W)
        x = self.proj(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 5. Residual connection
        return x + x_in
    

# ======================================================================
# 3. Core modules: Residual Block with Time Embedding
# ======================================================================

class ResBlock(nn.Module):
    


        



import torch
import torch.nn as nn
import torchvision.models as models

# Attention mechanism adapted for ResNet

class TransformerAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x_proj = self.proj(x)
        x_proj = self.drop(x_proj)
        
        qkv = torch.chunk(x_proj, chunks=3, dim=-1)
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3), qkv)
        
        sim = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(sim, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        out = self.norm2(out)
        out = self.ffn(out)
        return out + residual

# Residual block with feature squeezing and attention

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.transformer_attention = TransformerAttention(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Feature squeezing
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)

        # Apply transformer-based attention mechanism
        out = out.unsqueeze(-1)  # Add an extra dimension for the transformer
        out = self.transformer_attention(out)
        out = out.unsqueeze(-1)  # Restore the squeezed dimension

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetWithAttentionAndSqueeze(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        # Remove the original fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Modify the final layers to fit binary classification
        self.fc = nn.Linear(2048, num_classes)

        # Find appropriate blocks in the ResNet layers and add attention and feature squeezing
        for i in range(len(self.resnet)):
            if isinstance(self.resnet[i], nn.Sequential):
                for j in range(len(self.resnet[i])):
                    if isinstance(self.resnet[i][j], nn.Conv2d):
                        if self.resnet[i][j].out_channels == 1024:  # Add attention to the desired block
                            self.resnet[i][j] = ResidualAttentionBlock(1024, 2048, stride=2, downsample=None)
                        if self.resnet[i][j].out_channels == 2048:  # Add attention to another desired block
                            self.resnet[i][j] = ResidualAttentionBlock(2048, 2048, stride=2, downsample=None)
                            break  # Add attention to a single block and break the loop

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate the model
model = ResNetWithAttentionAndSqueeze()

# Print the model architecture
print(model)

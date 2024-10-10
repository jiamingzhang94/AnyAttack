import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super(EfficientAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels, head_count, value_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.attention = EfficientAttention(out_channels, key_channels, head_count, value_channels)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        residual = self.skip_conv(residual)
        out += residual
        return self.activation(out)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.activation(self.bn(self.conv(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim=1024, img_channels=3, img_size=224):
        super(Decoder, self).__init__()
        self.embedding_dim = embed_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 16  # Initial size before upsampling

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256 * self.init_size ** 2)
        )

        self.upsample_blocks = nn.ModuleList([
            ResBlock(256, 256, 64, 8, 256),
            UpBlock(256, 128),
            ResBlock(128, 128, 32, 8, 128),
            UpBlock(128, 64),
            ResBlock(64, 64, 16, 8, 64),
            UpBlock(64, 32),
            ResBlock(32, 32, 8, 8, 32),
            UpBlock(32, 16),
            ResBlock(16, 16, 4, 8, 16)
        ])

        self.final_conv = nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, embedding):
        embedding = embedding.view(embedding.size(0), -1)
        out = self.fc(embedding.float())
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        for block in self.upsample_blocks:
            out = block(out)
        img = self.final_conv(out)
        return img

    
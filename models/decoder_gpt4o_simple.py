import torch
import torch.nn as nn
import torch.nn.init as init


class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_conv = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.skip_conv(residual)
        out += residual
        return self.activation(out)


class PixelShuffleUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelShuffleUpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(self.bn(x))
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim=512, img_channels=3, img_size=224):
        super(SimpleDecoder, self).__init__()
        self.embedding_dim = embed_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 16  # Initial size before upsampling

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256 * self.init_size * self.init_size)
        )

        self.blocks = nn.ModuleList([
            SimpleResBlock(256, 256),
            PixelShuffleUpBlock(256, 128),
            SimpleResBlock(128, 128),
            PixelShuffleUpBlock(128, 64),
            SimpleResBlock(64, 64),
            PixelShuffleUpBlock(64, 32),
            SimpleResBlock(32, 32),
            PixelShuffleUpBlock(32, 16),
            SimpleResBlock(16, 16)
        ])

        self.final_conv = nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1)

        # Initialize learnable parameters a and b
        self.a = nn.Parameter(torch.ones(1, img_channels, self.img_size, self.img_size))
        self.b = nn.Parameter(torch.zeros(1, img_channels, self.img_size, self.img_size))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, embedding):
        embedding = embedding.view(embedding.size(0), -1)
        out = self.fc(embedding.float())
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)

        for block in self.blocks:
            out = block(out)

        img = self.final_conv(out)

        return img * self.a + self.b
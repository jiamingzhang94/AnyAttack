import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms


class CLIPEncoder(nn.Module):
    def __init__(self, model="RN50"):
        """
        CLIP Image Encoder using pre-trained CLIP models.

        Args:
            model (str): The model name. Supported models are:
                "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14-336",
                "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64".
        """
        super(CLIPEncoder, self).__init__()
        clip_model_names = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B/16": "openai/clip-vit-base-patch16",
            "ViT-L/14": "openai/clip-vit-large-patch14",
            "ViT-L/14-336": "openai/clip-vit-large-patch14-336",
            "RN50": "openai/clip-rn50",
            "RN101": "openai/clip-rn101",
            "RN50x4": "openai/clip-rn50x4",
            "RN50x16": "openai/clip-rn50x16",
            "RN50x64": "openai/clip-rn50x64"
        }
        assert model in clip_model_names, f"Model {model} is not supported. Supported models: {list(clip_model_names.keys())}"
        self.clip_model = CLIPModel.from_pretrained(clip_model_names[model])
        self.image_encoder = self.clip_model.vision_model
        self.text_encoder = self.clip_model.text_model
        self.processor = CLIPProcessor.from_pretrained(clip_model_names[model])

    def encode_img(self, images):
        """
        Forward pass for the CLIP image encoder.

        Args:
            images (torch.Tensor): A batch of images with shape (batch_size, 3, height, width).
                                   The images should be preprocessed and normalized.

        Returns:
            torch.Tensor: Image embeddings with shape (batch_size, embedding_dim).
        """
        assert images.ndim == 4 and images.shape[1] == 3, "Input images should have shape (batch_size, 3, height, width)"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = normalize(images)
        outputs = self.image_encoder(pixel_values=images)
        image_embeds = outputs.pooler_output
        return image_embeds

    def encode_text(self, inputs, device):
        """
        Forward pass for the CLIP text encoder.

        Args:
            texts (torch.Tensor): A batch of tokenized texts with shape (batch_size, seq_len).
                                  The texts should be preprocessed and tokenized.

        Returns:
            torch.Tensor: Text embeddings with shape (batch_size, embedding_dim).
        """
        texts = self.processor(text=inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.text_encoder(input_ids=texts['input_ids'], attention_mask=texts['attention_mask'])
        text_embeds = outputs.pooler_output
        return text_embeds


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class Decoder(nn.Module):
    def __init__(self, embed_dim=1024, latent_dim=256, num_res_blocks=3):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim * 7 * 7),
            nn.ReLU(inplace=True)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(latent_dim, 128, kernel_size=3, padding=1)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            self._make_up_block(128, 128),
            self._make_up_block(128, 64),
            self._make_up_block(64, 32),
            self._make_up_block(32, 16),
            self._make_up_block(16, 16),  # Added an extra upsampling block
            # self._make_up_block(16, 16)  # Added an extra upsampling block
        ])

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(16) for _ in range(num_res_blocks)
        ])

        # Self-attention layer
        self.self_attention = SelfAttention(16)

        # Final convolution
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_res_block(self, channels):
        return ResidualBlock(channels)

    def forward(self, x):
        # Project embedding to spatial representation
        x = x.view(x.size(0), -1)
        assert x.size(0) == 1024
        x = self.embed_proj(x)
        x = x.view(-1, self.latent_dim, 7, 7)

        # Initial convolution
        x = self.init_conv(x)

        # Upsampling
        for up_block in self.up_blocks:
            x = up_block(x)

        # Apply self-attention
        x = self.self_attention(x)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Final convolution
        x = self.final_conv(x)
        x = torch.tanh(x)  # Output in range [-1, 1]
        x = x*0.5 + 0.5

        return x




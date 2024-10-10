import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from mmpretrain import FeatureExtractor, get_model
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import argparse


class CLIP_Vision_encoder(nn.Module):
    def __init__(self, args):
        super(CLIP_Vision_encoder, self).__init__()
        self.args = args
        self.encoder = CLIPVisionModel.from_pretrained(args.clip_model_path)
        self.processer = CLIPProcessor.from_pretrained(self.args.clip_model_path)

    def forward(self, x):
        inputs = self.processer(images=x, return_tensors="pt")
        encode = self.encoder(**inputs, output_hidden_states=True, output_attentions=True)
        last_hidden_state = encode.last_hidden_state
        return last_hidden_state


class CLIP_Text_encoder(nn.Module):
    def __init__(self, args):
        super(CLIP_Text_encoder, self).__init__()
        self.args = args
        self.encoder = CLIPTextModel.from_pretrained(self.args.clip_model_path)
        self.Tokenizer = CLIPTokenizer.from_pretrained(self.args.clip_model_path)

    def forward(self, x):
        input = self.Tokenizer(x, padding=True, return_tensors="pt")
        encode = self.encoder(input_ids=input["input_ids"], output_hidden_states=True, output_attentions=True)
        last_hidden_state = encode.last_hidden_state
        return last_hidden_state


class Deconv(nn.Module):
    def __init__(self, input_chanel, output_chanel):
        super(Deconv,self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_chanel, output_chanel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class CLIP_encoder_decoder(nn.Module):
    def __init__(self, args):
        super(CLIP_encoder_decoder, self).__init__()
        self.args = args
        self.train=args.train
        if self.train:
            self.encoder = CLIP_Vision_encoder(args=args)
        else:
            self.encoder = CLIP_Text_encoder(args=args)
        # clip 编码得到的特征(batch_size,50,768)->(batch_size,49,768)
        self.linear = nn.Linear(50, 49)

        self.deconv5 = Deconv(768, 384)
        self.deconv4 = Deconv(384, 192)
        self.deconv3 = Deconv(192, 96)
        self.deconv2 = Deconv(96, 48)
        self.deconv1 = Deconv(48, 3)

    def forward(self, x):
        # 视觉编码器输出维度（batch_size,50,768）
        # 文本编码器输出维度（batch_size,10752,768）
        # 先实现视觉编码器部分
        feature = self.encoder(x)

        #  (batch_size, 50, 768) 转换为 (batch_size * 768, 50)
        reshaped_input = feature.permute(0, 2, 1).contiguous().view(-1, 50)
        center_feature = self.linear(reshaped_input)

        center_feature = center_feature.view(-1, 768, 7, 7)

        deconv_feature5 = self.deconv5(center_feature)
        deconv_feature4 = self.deconv4(deconv_feature5)
        deconv_feature3 = self.deconv3(deconv_feature4)
        deconv_feature2 = self.deconv2(deconv_feature3)
        image = self.deconv1(deconv_feature2)
        return image


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--train", type=str, default=True)
    argparse.add_argument("--clip_model_path", type=str, default="/data2/ModelWarehouse/clip-vit-base-patch32")
    args = argparse.parse_args()
    # encoder = CLIP_Vision_encoder(args=args)
    model = CLIP_encoder_decoder(args=args)
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # image = transform(image)
    # image = torch.randn(size=[5,3,224,224])
    # image="hello,hello,hello"
    print(model([image,image,image]).shape)
    # train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
from torchvision import transforms
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.ae_official import CLIPEncoder
from models.decoder_gpt4o import Decoder
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from lavis.datasets.builders import load_dataset
from lavis.common.registry import registry
import timm
import torchvision
from util import train_collate_fn
import argparse


class DirectMatchingLoss(nn.Module):
    def __init__(self):
        super(DirectMatchingLoss, self).__init__()

    def forward(self, image_features, text_features):
        cos_sim = torch.cosine_similarity(image_features, text_features, dim=-1)
        loss = -cos_sim.mean()

        return loss


class BiContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(BiContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        cos_sim = torch.matmul(image_features, text_features.t()) / self.temperature

        labels = torch.eye(cos_sim.size(0), device=cos_sim.device)

        loss_i2t = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
        loss_t2i = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()

        loss = (loss_i2t + loss_t2i) / 2

        avg_similarity = torch.diag(torch.matmul(image_features, text_features.t())).mean()

        return loss, avg_similarity


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def select_criterion(args, criterion, embed_adv, embed_tar):
    if args.criterion == 'BiContrastiveLoss':
        loss, observer = criterion(embed_adv, embed_tar)
    elif args.criterion == 'Cosine':
        loss = criterion(embed_adv, embed_tar)
        observer = -loss
    else:
        raise NameError
    return loss, observer


def make_dataloader(args, batch_size, rank, world_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    coco_dataset = load_dataset(args.dataset, vis_path=args.data_dir)
    train_dataset = coco_dataset['train']

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False,
                                   sampler=train_sampler, collate_fn=train_collate_fn, drop_last=True)

    imagenet_dataset = torchvision.datasets.ImageFolder(args.imagenet, transform=transform)
    imagenet_sampler = torch.utils.data.distributed.DistributedSampler(imagenet_dataset, num_replicas=world_size,
                                                                       rank=rank)

    data_loader_imagenet = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=4, pin_memory=True, drop_last=True, sampler=imagenet_sampler)
    return train_data_loader, data_loader_imagenet


def train(args):
    dist.init_process_group(backend='nccl')  # Initialize the process group for distributed training
    rank = dist.get_rank()  # Get the rank of the current process
    world_size = dist.get_world_size()  # Get total number of processes (GPUs)
    device = torch.device(f'cuda:{rank}')  # Set the device for the current process
    torch.cuda.set_device(device)  # Ensure that each process uses the correct GPU

    clip_encoder = CLIPEncoder('ViT-B/32').to(device)
    decoder = Decoder(embed_dim=512).to(device)

    # Models for auxiliary loss computation
    eva_encoder = timm.create_model("hf_hub:timm/eva02_large_patch14_448.mim_m38m_ft_in1k", num_classes=0,
                                    pretrained=True).to(device).eval()
    imagenet_encoder = torchvision.models.vit_b_16(pretrained=True).to(device).eval()
    imagenet_encoder.head = torch.nn.Identity()

    if args.checkpoint != 'scratch':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Wrap models with DistributedDataParallel (DDP)
    decoder = DDP(decoder, device_ids=[rank])

    # Optimizer, scheduler, and scaler
    optimizer = torch.optim.AdamW(decoder.parameters(), args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1)
    scaler = GradScaler()

    # Criterion based on args
    if args.criterion == 'BiContrastiveLoss':
        criterion = BiContrastiveLoss()
    elif args.criterion == 'Cosine':
        criterion = DirectMatchingLoss()
    else:
        raise ValueError

    train_loader, data_loader_imagenet = make_dataloader(args, args.batch_size, rank, world_size)
    data_loader_imagenet_cycle = cycle(data_loader_imagenet)

    start_epoch = 0
    for epoch in range(start_epoch, args.epoch):
        total_loss = 0
        total_observer_clip = 0
        total_observer_eva = 0
        total_observer_imagenet = 0
        global_step = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)

            with autocast():
                optimizer.zero_grad()

                with torch.no_grad():
                    embed_tar = clip_encoder.encode_img(images)  # CLIP encoder
                    images_ori, _ = next(data_loader_imagenet_cycle)

                    # Get target embeddings from auxiliary models
                    images_eva = F.interpolate(images, size=(448, 448), mode='bilinear')
                    embed_tar_eva = eva_encoder(images_eva)
                    embed_tar_imagenet = imagenet_encoder(images)

                noise = decoder(embed_tar)
                noise = torch.clamp(noise, -args.eps, args.eps)
                images_adv = torch.clamp(noise + images_ori.to(device), 0, 1)

                embed_adv = clip_encoder.encode_img(images_adv)  # CLIP adversarial embedding

                # Adversarial embeddings for auxiliary models
                images_adv_eva = F.interpolate(images_adv, size=(448, 448), mode='bilinear')
                embed_adv_eva = eva_encoder(images_adv_eva)
                embed_adv_imagenet = imagenet_encoder(images_adv)

                # Compute all three losses and observers
                loss, observer_clip = select_criterion(args, criterion, embed_adv, embed_tar)
                loss_eva, observer_eva = select_criterion(args, criterion, embed_adv_eva, embed_tar_eva)
                loss_imagenet, observer_imagenet = select_criterion(args, criterion, embed_adv_imagenet,
                                                                    embed_tar_imagenet)

                total_loss_combined = loss + loss_eva + loss_imagenet  # Combine the three losses

                scaler.scale(total_loss_combined).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += total_loss_combined.item()
                total_observer_clip += observer_clip.item()
                total_observer_eva += observer_eva.item()
                total_observer_imagenet += observer_imagenet.item()
                global_step += 1
                scheduler.step()

            if batch_idx % 100 == 0 and rank == 0:
                avg_loss = total_loss / global_step
                avg_observer_clip = total_observer_clip / global_step
                avg_observer_eva = total_observer_eva / global_step
                avg_observer_imagenet = total_observer_imagenet / global_step
                current_lr = optimizer.param_groups[0]['lr']

                # Print losses and observers for each model
                print(
                    f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.6f}, CLIP Similarity: {avg_observer_clip:.4f},'
                    f'EVA Similarity: {avg_observer_eva:.4f}, ImageNet Similarity: {avg_observer_imagenet:.4f}, lr: {current_lr}')

        avg_loss = total_loss / global_step
        if rank == 0:
            print(f"Epoch: {epoch}, Training Loss: {avg_loss}")

        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'decoder_state_dict': decoder.module.state_dict(),  # Saving DDP-wrapped model
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f"checkpoints/{args.dataset}_{args.criterion}_auxiliary.pt")

    dist.destroy_process_group()  # Clean up after training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/mscoco")
    parser.add_argument("--dataset", type=str, default="coco_retrieval")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=800)
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default='checkpoints/pre-trained.pt', help="path to checkpoint to load")
    parser.add_argument("--criterion", type=str, default='BiContrastiveLoss')
    parser.add_argument("--imagenet", type=str, default='datasets/ILSVRC2012/val')
    parser.add_argument("--cache_path", type=str, default='datasets')
    args = parser.parse_args()

    registry.mapping["paths"]["cache_root"] = args.cache_path

    train(args)

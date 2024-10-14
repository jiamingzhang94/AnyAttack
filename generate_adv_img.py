import tqdm
from torch.utils.data import Dataset
import torchvision
from models.model import CLIPEncoder, Decoder
from dataset import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=16./255)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument("--clean_image_path", type=str)
    parser.add_argument("--target_caption", type=str)
    parser.add_argument("--target_image_path", type=str)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str,
                        default="outputs")
    parser.add_argument("--adv_imgs", type=str)
    parser.add_argument("--dataset", type=str, help='coco flickr30k')
    args = parser.parse_args()

    device = args.device
    # model
    print(f"Loading CLIP models: {args.model_name}...")
    clip_model = CLIPEncoder('ViT-B/32').to(device)
    print(f"Loading Decoder: {args.decoder_path.split('/')[-1]}...")
    decoder = Decoder(embed_dim=512).to(device).eval()
    try:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu')["decoder_state_dict"])
    except:
        state_dict = torch.load(args.decoder_path, map_location='cpu')["decoder_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        decoder.load_state_dict(new_state_dict)

    imagenet_data = ImageFolder(args.clean_image_path, transform=eval_transform)
    target_data = ImageTextDataset(args.target_caption, args.target_image_path, transform=eval_transform)

    data_loader_imagenet = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=8)
    data_loader_target = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=8)

    adv_tensor = []
    img_id = 0
    adv_data = []

    total_sim_emb = 0
    total_batches = 0

    for idx, ((clean_image, label), (target_image, text)) in tqdm.tqdm(enumerate(zip(data_loader_imagenet, data_loader_target))):
        clean_image = clean_image.to(device)
        target_image = target_image.to(device)

        with torch.no_grad():

            img_emb = clip_model.encode_img(target_image)
            target_emb = img_emb
            origin_noise = decoder(img_emb)
            noise = torch.clamp(origin_noise, -args.eps, args.eps)
            adv_image = clean_image + noise
            adv_image = torch.clamp(adv_image, 0, 1)
            total_batches += 1

        # save images
        adv_image_path = os.path.join(args.output_path, args.dataset, args.adv_imgs)
        if not os.path.exists(adv_image_path):
            os.makedirs(adv_image_path)
        for i in range(adv_image.shape[0]):
            torchvision.utils.save_image(adv_image[i], os.path.join(adv_image_path, f"{img_id:05d}.") + 'png')
            if args.dataset == "snli_ve":
                adv_data.append(
                    {
                        'image': f"{img_id:05d}.png",
                        'caption': [text[i]],
                        "sentence": text[i],
                        "label": "entailment"
                    }
                )
            else:
                adv_data.append(
                    {
                        'image': f"{img_id:05d}.png",
                        'caption':[text[i]]
                    }
                )
            img_id += 1

    print(f"Saved adv images: {args.dataset}/{args.adv_imgs}")


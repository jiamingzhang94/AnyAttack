import argparse
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from models.model import CLIPEncoder, Decoder


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=16. / 255)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--decoder_path", type=str, required=True)
    parser.add_argument("--clean_image_path", type=str, required=True)
    parser.add_argument("--target_image_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default="output.png")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load models
    print(f"Loading CLIP model: {args.model_name}...")
    clip_model = CLIPEncoder(args.model_name).to(device)

    print(f"Loading Decoder: {args.decoder_path.split('/')[-1]}...")
    decoder = Decoder(embed_dim=512).to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu')["decoder_state_dict"])

    # Load images
    clean_image = load_image(args.clean_image_path).to(device)
    target_image = load_image(args.target_image_path).to(device)

    # Generate noise
    with torch.no_grad():
        img_emb = clip_model.encode_img(target_image)
        origin_noise = decoder(img_emb)
        noise = torch.clamp(origin_noise, -args.eps, args.eps)
        adv_image = clean_image + noise
        adv_image = torch.clamp(adv_image, 0, 1)

    # Save output image
    torchvision.utils.save_image(adv_image[0], args.output_path)
    print(f"Saved adversarial image to: {args.output_path}")


if __name__ == "__main__":
    main()
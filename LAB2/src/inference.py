import argparse
import os
import torch
from torch.utils.data import DataLoader

from oxford_pet import load_dataset
from models.unet import UNetModel
from models.resnet34_unet import ResNet34_UNet
from utils import compute_dice_score

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.model == "unet.pth":
        model = UNetModel(in_channels=3, out_channels=1).to(device)
    elif args.model == "resnet34_unet.pth":
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Unsupported model file")
    
    model_path = os.path.join("..", "saved_models", args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded state dict from {model_path}\n(inference on unet model.)")
    
    model.eval()
    
    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    total_dice = 0.0
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(images)

            for i in range(images.size(0)):
                dice = compute_dice_score(preds[i], masks[i])
                total_dice += dice
                count += 1
    
    mean_dice = total_dice / count if count > 0 else 0.0
    print(f"Mean Dice Score on test set: {mean_dice:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description='Test model and compute mean Dice score')
    parser.add_argument('--model', default='unet.pth', choices=["unet.pth", "resnet34_unet.pth"],
                        help='Name of the model file in ../saved_models directory')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet/",
                        help='Path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for testing')
    return parser.parse_args()

if __name__ == '__main__':
    main()

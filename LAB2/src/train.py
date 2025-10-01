import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNetModel
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from utils import compute_dice_score, compute_loss
from evaluate import evaluate

def train(args):
    # Load datasets
    train_dataset = load_dataset(args.data_path, mode="train")
    valid_dataset = load_dataset(args.data_path, mode="valid")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # Build model
    if args.model == "unet":
        model = UNetModel(in_channels=3, out_channels=1).to(args.device)
    else:
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_dice = 0.0  # Track the best validation Dice score

    # Create a list to record performance per epoch
    performance_records = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses, epoch_dices = [], []
        epoch_start_time = time.time()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar}')
        
        for i, batch in progress_bar:
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = compute_loss(outputs, masks, loss_type=args.loss_type)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            with torch.no_grad():
                epoch_dices.append(compute_dice_score(outputs, masks))
            
            progress_bar.set_description(
                f"Epoch {epoch}/{args.epochs} | Loss: {loss.item():.4f} | Dice: {epoch_dices[-1]:.4f} | Iter: {i+1}/{len(train_loader)}"
            )
        
        # Evaluate on validation set
        val_losses, val_dices = evaluate(model, valid_loader, args.device, loss_type=args.loss_type)
        avg_train_loss = np.mean(epoch_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_dice = np.mean(epoch_dices)
        avg_val_dice = np.mean(val_dices)
        
        print(f"Epoch {epoch} => Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} | Valid Loss: {avg_val_loss:.4f}, Valid Dice: {avg_val_dice:.4f}")
        
        # Record performance for current epoch
        performance_records.append({
            "Epoch": epoch,
            "Train Loss": avg_train_loss,
            "Train Dice": avg_train_dice,
            "Valid Loss": avg_val_loss,
            "Valid Dice": avg_val_dice
        })

        # Save model if validation dice score > 0.9 and improved
        if avg_val_dice > 0.9 and avg_val_dice > best_dice:
            best_dice = avg_val_dice
            save_dir = os.path.join("..", "saved_models")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, f"{args.model}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path} with Validation Dice Score: {best_dice:.4f}")

    # Ensure outputs folder exists
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    # Convert performance records to DataFrame and rename columns based on model type
    df_new = pd.DataFrame(performance_records)
    if args.model == "unet":
        new_columns = {
            "Train Loss": "Unet Train Loss",
            "Train Dice": "Unet Train Dice",
            "Valid Loss": "Unet Val Loss",
            "Valid Dice": "Unet Val Dice"
        }
        desired_columns = ["Epoch", "Unet Train Loss", "Unet Train Dice", "Unet Val Loss", "Unet Val Dice"]
    else:
        new_columns = {
            "Train Loss": "ResNet34_Unet Train Loss",
            "Train Dice": "ResNet34_Unet Train Dice",
            "Valid Loss": "ResNet34_Unet Val Loss",
            "Valid Dice": "ResNet34_Unet Val Dice"
        }
        desired_columns = ["Epoch", "ResNet34_Unet Train Loss", "ResNet34_Unet Train Dice", "ResNet34_Unet Val Loss", "ResNet34_Unet Val Dice"]

    df_new = df_new.rename(columns=new_columns)
    df_new.set_index("Epoch", inplace=True)
    
    output_excel = os.path.join(outputs_dir, "model_performence.xlsx")

    if os.path.exists(output_excel):
        df_existing = pd.read_excel(output_excel)
        df_existing.set_index("Epoch", inplace=True)
        # Overwrite columns corresponding to the current model
        for col in new_columns.values():
            df_existing[col] = df_new[col]
        # Also add new epochs if any exist in df_new but not in df_existing
        df_merged = df_existing.combine_first(df_new)
    else:
        df_merged = df_new

    df_merged.reset_index(inplace=True)
    df_merged = df_merged[desired_columns]
    
    df_merged.to_excel(output_excel, index=False)
    print(f"Performance records saved to {output_excel}")

def get_args():
    parser = argparse.ArgumentParser(description="Train segmentation model using BCE + Dice/Tversky Loss")
    parser.add_argument('--model', type=str, default="unet", choices=["unet", "resnet34_unet"],
                        help="Model to use: unet or resnet34_unet")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for training")
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet/",
                        help="Path to the dataset")
    parser.add_argument('--epochs', '-e', type=int, default=200, help="Number of epochs")
    parser.add_argument('--batch_size', '-b', type=int, default=32, help="Batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--loss_type', type=str, default="bce_dice", choices=["bce_dice", "bce_tversky"],
                        help="Loss combination to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)

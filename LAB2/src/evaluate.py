import numpy as np
import torch
import torch.nn as nn
from utils import compute_dice_score, compute_loss

def evaluate(model: nn.Module, dataloader, device: torch.device, loss_type: str = 'bce_dice'):
    model.eval()
    losses, dice_scores = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            pred_masks = model(images)
            loss = compute_loss(pred_masks, masks, loss_type=loss_type)
            losses.append(loss.item())
            dice_scores.append(compute_dice_score(pred_masks, masks))
    
    avg_loss = np.mean(losses)
    avg_dice = np.mean(dice_scores)
    
    return losses, dice_scores

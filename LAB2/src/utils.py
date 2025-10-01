import torch
import torch.nn as nn
import os
from torchvision.utils import save_image

def compute_dice_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    binarized_pred = (pred_mask > threshold).float()
    intersection = (binarized_pred * gt_mask).sum()
    total_pixels = binarized_pred.sum() + gt_mask.sum()
    dice = 2.0 * intersection / (total_pixels + eps)
    return dice.item()

def compute_dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    intersection = torch.sum(pred_mask * gt_mask) + eps
    union = torch.sum(pred_mask) + torch.sum(gt_mask) + eps
    loss = 1 - (2 * intersection / union)
    return loss

def compute_tversky_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred_mask.view(-1)
    gt_flat = gt_mask.view(-1)
    TP = (pred_flat * gt_flat).sum()
    FP = ((1 - gt_flat) * pred_flat).sum()
    FN = (gt_flat * (1 - pred_flat)).sum()
    tversky_index = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return 1 - tversky_index

def compute_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, loss_type: str = 'bce_dice') -> torch.Tensor:
    bce = nn.BCELoss()(pred_mask, gt_mask)
    if loss_type == 'bce_dice':
        extra_loss = compute_dice_loss(pred_mask, gt_mask)
    elif loss_type == 'bce_tversky':
        extra_loss = compute_tversky_loss(pred_mask, gt_mask)
    else:
        raise ValueError("Unsupported loss_type")
    return bce + extra_loss

def visualize_segmentation(image: torch.Tensor, pred_mask: torch.Tensor, gt_mask: torch.Tensor, file_name: str) -> None:
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    restored_img = image.clone()
    restored_img[0] = restored_img[0] * std[0] + mean[0]
    restored_img[1] = restored_img[1] * std[1] + mean[1]
    restored_img[2] = restored_img[2] * std[2] + mean[2]
    
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)
    if gt_mask.dim() == 2:
        gt_mask = gt_mask.unsqueeze(0)

    pred_mask_rgb = pred_mask.repeat(3, 1, 1)
    gt_mask_rgb = gt_mask.repeat(3, 1, 1)

    masked_pred_img = restored_img * pred_mask_rgb
    masked_gt_img = restored_img * gt_mask_rgb

    concatenated = torch.cat([restored_img, pred_mask_rgb, gt_mask_rgb, masked_pred_img, masked_gt_img], dim=2)

    os.makedirs("test_img", exist_ok=True)
    save_image(concatenated, f"test_img/{file_name}.png")
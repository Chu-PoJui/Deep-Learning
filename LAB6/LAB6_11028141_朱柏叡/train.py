import os
import sys

# Ensure we can import dataset from the same folder
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

from dataset import DiffusionDataset

class ClassCondUNet(nn.Module):
    def __init__(self,
                 num_classes=24,
                 class_emb_size=512,
                 blocks=[0,0,0,0,0,0],
                 channels=[1,1,2,2,4,4],
                 img_size=128):
        super().__init__()
        first_ch = class_emb_size // 4
        down = ["DownBlock2D" if b==0 else "AttnDownBlock2D" for b in blocks]
        up   = ["UpBlock2D"   if b==0 else "AttnUpBlock2D"   for b in reversed(blocks)]
        chs  = [first_ch * c for c in channels]

        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=tuple(chs),
            down_block_types=tuple(down),
            up_block_types=tuple(up),
        )
        # replace class_embedding
        self.unet.class_embedding = nn.Linear(num_classes, class_emb_size)

    def forward(self, x, t, y):
        return self.unet(x, t, y).sample  # predict noise

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('ckpt', exist_ok=True)

    # Paths
    latest_ckpt = 'ckpt/latest.pth'

    # Dataset & DataLoader
    ds = DiffusionDataset(
        img_root='../iclevr',
        ann_file='train.json',
        objects_file='objects.json',
        high_res=128
    )
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Model, scheduler, optimizer, loss
    model = ClassCondUNet().to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2'
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    mse = nn.MSELoss()

    # Resume weights if exists
    if os.path.exists(latest_ckpt):
        state_dict = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        print(f"▷ Loaded model weights from {latest_ckpt}")

    # Training loop
    for epoch in range(0, 200):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        epoch_losses = []
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            noise = torch.randn_like(imgs)
            timesteps = torch.randint(0, 1000, (imgs.size(0),), device=device)

            noisy = scheduler.add_noise(imgs, noise, timesteps)
            pred  = model(noisy, timesteps, labels)
            loss  = mse(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{sum(epoch_losses)/len(epoch_losses):.4f}")

        # Save model.state_dict() every 20 epochs and at final epoch
        if epoch % 20 == 0 or epoch == 199:
            ckpt_path = f"ckpt/epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), latest_ckpt)
            print(f"▷ Saved model.state_dict() to {ckpt_path} and updated {latest_ckpt}")

if __name__ == '__main__':
    main()

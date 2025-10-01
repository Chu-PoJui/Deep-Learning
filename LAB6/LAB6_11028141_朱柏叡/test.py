import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm

from dataset import DiffusionDataset
from evaluator import evaluation_model
from train import ClassCondUNet

def evaluate(ann_file, ckpt_path, out_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load only the test dataset: mode='test'
    ds = DiffusionDataset(
        img_root=None,             
        ann_file=ann_file,
        objects_file='objects.json',
        high_res=128,
        mode='test'
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 2) Build the model and load state_dict
    model = ClassCondUNet().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # 3) scheduler & evaluator
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2'
    )
    ev = evaluation_model()

    results, accs = [], []
    for labels in tqdm(loader, desc=f"Eval {ann_file}"):
        # The loader returns a hot
        labels = labels.to(device)      # shape (1,24)
        x = torch.randn(1,3,128,128, device=device)

        # DDPM reverse sampling loop
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(x, t, labels)
            x = scheduler.step(noise_pred, t, x).prev_sample

        # Downsample to 64×64
        x64 = F.interpolate(x, size=(64,64),
                            mode='bilinear', align_corners=False)
        results.append(x64.cpu().squeeze(0))

        # evaluate
        accs.append(ev.eval(x64, labels))

    # Save the grid image
    grid = torchvision.utils.make_grid(results, nrow=8,
                                       normalize=True, padding=2)
    img = T.ToPILImage()(grid)
    img.save(os.path.join(out_dir, 'grid.png'))

    mean_acc = sum(accs)/len(accs) if accs else 0.0
    print(f"{ann_file} → Mean Accuracy: {mean_acc:.4f}")

if __name__ == '__main__':
    # Execute these two lines in the project root directory
    evaluate('test.json',    'ckpt/epoch200.pth', 'results/test')
    evaluate('new_test.json',  'ckpt/epoch200.pth', 'results/new_test')

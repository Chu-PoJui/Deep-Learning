import os, json, torch, torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from diffusers import DDPMScheduler
from train import ClassCondUNet  # Make sure train.py is in the same directory

def generate_denoise_process(label_names, 
                             ckpt_path='ckpt/epoch200.pth', 
                             out_path='denoise_process.png',
                             high_res=128, low_res=64,
                             capture_steps=(999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1. Load label→idx mapping
    with open('objects.json','r') as f:
        obj2idx = json.load(f)
    # 2. Build one-hot
    y = torch.zeros(1, len(obj2idx), device=device)
    for n in label_names:
        y[0, obj2idx[n]] = 1

    # 3. Build the model and load weights
    model = ClassCondUNet().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # 4. Build a scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2'
    )

    # 5. Randomly initialize high-resolution noise
    x = torch.randn(1, 3, high_res, high_res, device=device)

    # 6. Iterate denoising and record at key steps
    imgs = []
    for t in scheduler.timesteps:
        with torch.no_grad():
            pred_noise = model(x, t, y)
        x = scheduler.step(pred_noise, t, x).prev_sample

        if t in capture_steps:
            # Downsampling to lower resolution
            x_low = F.interpolate(x, size=(low_res, low_res),
                                  mode='bilinear', align_corners=False)
            imgs.append(x_low.cpu().squeeze(0))

    # 7. Puzzle and save
    grid = make_grid(imgs, nrow=len(imgs), normalize=True, padding=2)
    save_image(grid, out_path)
    print(f"▷ Saved denoising process to {out_path}")

if __name__ == '__main__':
    # ["red sphere","cyan cylinder","cyan cube"] 
    generate_denoise_process(
        ["red sphere","cyan cylinder","cyan cube"],
        ckpt_path='ckpt/epoch200.pth',
        out_path='denoise_process.png'
    )

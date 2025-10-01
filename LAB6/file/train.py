import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ICLEVR
from ddpm    import ConditionalDDPM

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json_root', default='.', help='含 JSON 與 objects.json 的資料夾')
    p.add_argument('--img_root',  default='../iclevr', help='訓練圖的資料夾')
    p.add_argument('--epochs',    type=int,   default=40)
    p.add_argument('--bs',        type=int,   default=256)
    p.add_argument('--lr',        type=float, default=2e-4)
    p.add_argument('--device',    default='cuda')
    p.add_argument('--save_dir',  default='checkpoints')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ds = ICLEVR(args.json_root, mode="train", img_root=args.img_root)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    num_workers=4, pin_memory=True)

    model = ConditionalDDPM(device=args.device).to(args.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        losses=[]
        pbar = tqdm(dl, desc=f"Epoch {ep}")
        for x,y in pbar:
            x,y = x.to(args.device), y.to(args.device)
            l   = model.loss(x,y)
            opt.zero_grad(); l.backward(); opt.step()
            losses.append(l.item())
            pbar.set_postfix(loss=np.mean(losses))
        torch.save({'model':model.state_dict()}, f"{args.save_dir}/ckpt_e{ep}.pth")
        torch.cuda.empty_cache()

if __name__=="__main__":
    main()

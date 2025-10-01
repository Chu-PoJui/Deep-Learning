# file/test.py
import os, json, argparse, tqdm, torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
from diffusers import DDIMScheduler

from ddpm      import ConditionalDDPM
from evaluator import evaluation_model

def txt2onehot(objs,obj2idx):
    v = torch.zeros(len(obj2idx))
    for o in objs: v[obj2idx[o]] = 1
    return v

@torch.no_grad()
def generate(model, sched, conds, obj2idx, outdir, steps, scale, device):
    """
    1) 逐张生成并保存到 outdir/00.png ... 31.png
    2) 把所有小图合成一张 8x4 网格并保存为 outdir_grid.png
    """
    os.makedirs(outdir, exist_ok=True)
    imgs = []
    for i, objs in enumerate(tqdm.tqdm(conds, desc=outdir)):
        y  = txt2onehot(objs,obj2idx)[None].to(device)
        x0 = model.sample(y, dd_steps=steps, guidance_scale=scale)
        imgs.append(x0[0].cpu())
        # 保存单张
        save_image((x0 + 1) / 2, f"{outdir}/{i:02d}.png")
    # 合成网格图
    grid = make_grid(imgs, nrow=8, normalize=True, value_range=(-1,1))
    save_image(grid, f"{outdir}_grid.png")
    return imgs

def eval_folder(folder, conds, evaluator, obj2idx):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    import glob
    # 读回小图并评分类器
    imgs = torch.stack([
        tf(Image.open(f).convert("RGB"))
        for f in sorted(glob.glob(f"{folder}/*.png"))
    ]).cuda()
    labels = torch.stack([txt2onehot(o,obj2idx) for o in conds]).cuda()
    return evaluator.eval(imgs, labels)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json_root', default='.', help='含 JSON 与 objects.json 的目录')
    p.add_argument('--ckpt',      required=True, help='ckpt 文件路径')
    p.add_argument('--steps',     type=int,   default=50,  help='DDIM 步数')
    p.add_argument('--scale',     type=float, default=4.0, help='guidance scale')
    p.add_argument('--device',    default='cuda')
    args = p.parse_args()

    device = args.device
    # 1) load model
    model = ConditionalDDPM(device=device).to(device)
    model.load_state_dict(torch.load(args.ckpt)['model'])
    model.eval()

    # 2) build DDIM scheduler
    sched = DDIMScheduler.from_config(model.scheduler.config)
    sched.set_timesteps(args.steps)

    # 3) load conditions
    with open(os.path.join(args.json_root, "objects.json")) as f:
        obj2idx = json.load(f)
    test_conds    = json.load(open(os.path.join(args.json_root, "test.json")))
    new_conds     = json.load(open(os.path.join(args.json_root, "new_test.json")))

    # 4) generate + save grid
    imgs1 = generate(model, sched, test_conds, obj2idx,
                     outdir="gen_test", steps=args.steps,
                     scale=args.scale, device=device)
    imgs2 = generate(model, sched, new_conds, obj2idx,
                     outdir="gen_new_test", steps=args.steps,
                     scale=args.scale, device=device)

    # 如果还需要把两组各自的图合并为一张 64 格大图，可参考：
    all_imgs = imgs1 + imgs2
    big_grid = make_grid(all_imgs, nrow=8, normalize=True, value_range=(-1,1))
    save_image(big_grid, "all_64_grid.png")

    # 5) 可视化去噪过程
    demo = txt2onehot(["red sphere","cyan cylinder","cyan cube"], obj2idx)[None]
    model.vis(demo, guidance_scale=args.scale, out="denoise_process.png")

    # 6) 评估 Accuracy
    evaluator = evaluation_model()
    acc1 = eval_folder("gen_test",     test_conds, evaluator, obj2idx)
    acc2 = eval_folder("gen_new_test", new_conds, evaluator, obj2idx)
    print(f"test acc: {acc1:.4f}, new_test acc: {acc2:.4f}, avg: {(acc1+acc2)/2:.4f}")

if __name__=="__main__":
    main()

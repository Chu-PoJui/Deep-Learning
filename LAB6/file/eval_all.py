# eval_all.py
import os, json, torch, argparse, tqdm
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
def generate_and_eval(model, sched, conds, obj2idx, evaluator, steps, scale, device):
    # 直接只算最后一张的 accuracy
    # (不存图)
    y  = torch.stack([txt2onehot(o,obj2idx) for o in conds]).to(device)
    x0 = model.sample(y, dd_steps=steps, guidance_scale=scale)
    # normalize for evaluator
    imgs = (x0.clamp(-1,1) + 1) / 2
    # to classifier norm
    tf = transforms.Compose([transforms.Normalize((0.5,)*3,(0.5,)*3)])
    imgs = imgs * 2 - 1  # back to [-1,1]
    return evaluator.eval(imgs, y.cpu())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json_root', default='.', help='JSON 及 objects.json 所在目录')
    p.add_argument('--ckpt_dir',  default='checkpoints', help='存放 ckpt_e*.pth 的目录')
    p.add_argument('--steps',     type=int,   default=50)
    p.add_argument('--scale',     type=float, default=4.0)
    p.add_argument('--device',    default='cuda')
    args = p.parse_args()

    # load conditions & evaluator once
    with open(os.path.join(args.json_root, "objects.json")) as f:
        obj2idx = json.load(f)
    test_conds   = json.load(open(os.path.join(args.json_root, "test.json")))
    new_conds    = json.load(open(os.path.join(args.json_root, "new_test.json")))
    evaluator = evaluation_model()

    # collect ckpts
    ckpts = [fn for fn in os.listdir(args.ckpt_dir) if fn.startswith("ckpt_e") and fn.endswith(".pth")]
    ckpts = sorted(ckpts, key=lambda x: int(x.replace("ckpt_e","").replace(".pth","")))

    results = []
    for ckpt in ckpts:
        epoch = int(ckpt.replace("ckpt_e","").replace(".pth",""))
        path  = os.path.join(args.ckpt_dir, ckpt)

        # load model
        model = ConditionalDDPM(device=args.device).to(args.device)
        model.load_state_dict(torch.load(path)['model'])
        model.eval()

        # scheduler
        sched = DDIMScheduler.from_config(model.scheduler.config)
        sched.set_timesteps(args.steps)

        # eval test / new_test
        acc1 = generate_and_eval(model,sched,test_conds, obj2idx, evaluator, args.steps, args.scale, args.device)
        acc2 = generate_and_eval(model,sched,new_conds,  obj2idx, evaluator, args.steps, args.scale, args.device)
        avg  = (acc1 + acc2) / 2
        print(f"Epoch {epoch:02d}: test={acc1:.4f}, new_test={acc2:.4f}, avg={avg:.4f}")
        results.append((epoch, acc1, acc2, avg))

    # optionally save to csv
    import csv
    with open("eval_results.csv","w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["epoch","test_acc","new_test_acc","avg_acc"])
        for row in results: w.writerow(row)

if __name__=="__main__":
    main()

import torch, torch.nn.functional as F, torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

def expand_labels(y, size=64):
    # y: (B,24) → (B,24,size,size)
    return y[:,:,None,None].expand(-1,-1,size,size)

class ConditionalDDPM(nn.Module):
    def __init__(self,
                 img_size=64,
                 timesteps=1000,
                 p_uncond=0.1,
                 device="cuda"):
        super().__init__()
        self.device   = device
        self.p_uncond = p_uncond

        # UNet with 3 + 24 channels
        self.model = UNet2DModel(
            sample_size        = img_size,
            in_channels        = 3+24,
            out_channels       = 3,
            block_out_channels = (64,128,256,256),
            down_block_types   = ("DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
            up_block_types     = ("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
        ).to(device)

        # scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps = timesteps,
            beta_schedule       = "linear",
            prediction_type     = "epsilon",
        )

    def loss(self, x0, y):
        B = x0.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps,(B,),device=self.device)
        noise = torch.randn_like(x0)
        xt = self.scheduler.add_noise(x0, noise, t)

        # classifier-free dropout
        mask = (torch.rand(B,1,device=self.device) < self.p_uncond)
        y_cf = y.masked_fill(mask,0)

        # concat
        y_map = expand_labels(y_cf)
        inp = torch.cat([xt, y_map],1)
        pred = self.model(inp, t).sample
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, y, dd_steps=50, guidance_scale=4.0):
        N = y.shape[0]
        self.scheduler.set_timesteps(dd_steps)
        x = torch.randn(N,3,64,64,device=self.device)

        y_map = expand_labels(y.to(self.device))
        zero_map = torch.zeros_like(y_map)
        for t in self.scheduler.timesteps:
            eps_c = self.model(torch.cat([x,y_map],1), t).sample
            eps_u = self.model(torch.cat([x,zero_map],1), t).sample
            eps   = eps_u + guidance_scale*(eps_c-eps_u)
            x     = self.scheduler.step(eps, t, x).prev_sample
        return x.clamp(-1,1)

    @torch.no_grad()
    def vis(self, y, steps=(49,40,30,20,10,0), guidance_scale=4.0, out="denoise.png"):
        x = torch.randn(1,3,64,64,device=self.device)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        y_map = expand_labels(y.to(self.device))
        zero_map = torch.zeros_like(y_map)
        imgs=[]
        for t in self.scheduler.timesteps:
            eps_c = self.model(torch.cat([x,y_map],1), t).sample
            eps_u = self.model(torch.cat([x,zero_map],1), t).sample
            eps   = eps_u + guidance_scale*(eps_c-eps_u)
            x     = self.scheduler.step(eps, t, x).prev_sample
            if t in steps: imgs.append(x.squeeze(0).cpu())
        from torchvision.utils import make_grid, save_image
        grid = make_grid(imgs, nrow=len(imgs), normalize=True, value_range=(-1,1))
        save_image(grid, out)

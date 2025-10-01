import torch, gymnasium as gym, numpy as np, argparse
from dqn import MLP_DQN, IdentityPreprocessor          # 直接複用你的類別

def main(path, episodes, render):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    model = MLP_DQN(4, 2).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    pre = IdentityPreprocessor()

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(); state = pre.reset(obs)
        done = False; total = 0
        while not done:
            with torch.no_grad():
                act = model(torch.from_numpy(state).float().unsqueeze(0).to(device)).argmax().item()
            nxt, r, term, trunc, _ = env.step(act)
            done, total = term or trunc, total + r
            state = pre.step(nxt)
        rewards.append(total); print(f"ep{ep:03d}  {total}")
    print(f"Mean {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--eps", type=int, default=20)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    main(args.model, args.eps, args.render)
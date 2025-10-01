import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

from dqn import DuelingDQN, AtariPreprocessor
        
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    num_actions = env.action_space.n

    # 1) 用訓練時同樣的網路
    model = DuelingDQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 2) 前處理也同樣用 AtariPreprocessor + LazyFrames
    processor = AtariPreprocessor(4)

    os.makedirs(args.output_dir, exist_ok=True)

    rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed+ep)
        state = processor.reset(obs)    # 回傳 LazyFrames
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_np = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            state = processor.step(next_obs)
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")

        rewards.append(total_reward)

    mean, std = np.mean(rewards), np.std(rewards)
    print(f"\nMean over {ep} eps → {mean:.1f} ± {std:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task3")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)

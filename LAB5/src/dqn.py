# -*- coding: utf-8 -*-
"""
`dqn.py` also supports:
# Task 1 - CartPole-v1 (Vector Observation, Vanilla DQN)
# Task 2 - Pong‑v5 (Image Observation, Natural CNN DQN)
# Task 3 - Pong‑v5 (Image Observation, Enhanced DQN: Dual DQN + Dueling + Prioritized ER + Multi-step Retrieval + LazyFrames)

Training Methods:
```
# Task 1 – CartPole
python .\LAB5_11028141_朱柏叡_Code\dqn.py 1 --env-name CartPole-v1 --lr 5e-4 --eps-decay 0.9999 --target-freq 1000 --replay-start 1000 --episodes 3000 --batch-size 64 --memory-size 100000

# Task 2 – Original Pong DQN
python .\LAB5_11028141_朱柏叡_Code\dqn.py 1 --env-name ALE/Pong-v5 --lr 1e-4 --eps-decay 0.999995  --replay-start 50000 --episodes 20000  --memory-size 90000

# Task 3 – Strengthening Pong DQN
python .\LAB5_11028141_朱柏叡_Code\dqn.py 3 --env ALE/Pong-v5
```

Evaluating Methods：
```
# Task 1 – CartPole
python .\LAB5_11028141_朱柏叡_Code\test_model_task1.py  --model .\LAB5_11028141_task1_cartpole.pt --render
(Mean 500.0 ± 0.0)

# Task 2 – Original Pong DQN
python .\LAB5_11028141_朱柏叡_Code\test_model_task2.py  --model-path .\LAB5_11028141_task2_pong.pt --episodes 20
(Mean over 20 eps → 19.0 ± 1.8)

# Task 3 – Strengthening Pong DQN
python .\LAB5_11028141_朱柏叡_Code\test_model_task3.py  --model-path .\LAB5_11028141_task3_pong800000.pt --episodes 20

```

Analysis diagram links (WANDB)：
```
# Task 1 – CartPole
https://wandb.ai/david8899b-chung-yuan-christian-university/DLP-Lab5-DQN-CartPole/runs/f6ujfsxl?nw=nwuserdavid8899b

# Task 2 – Original Pong DQN
https://wandb.ai/david8899b-chung-yuan-christian-university/DLP-Lab5-DQN-Pong/runs/3o9sgne7?nw=nwuserdavid8899b

# Task 3 – Strengthening Pong DQN
https://wandb.ai/david8899b-chung-yuan-christian-university/DLP-Lab5-task3/runs/471gg238?nw=nwuserdavid8899b

```
Automatically select tasks: preprocessing, network, replay buffer, and training process.
Wandb projects are also automatically named according to the tasks.
"""

import argparse, os, random, time
from collections import deque, namedtuple
from typing import Deque, Tuple, List

# ---------- imports ----------
import gymnasium as gym
import ale_py                  # Atari Environment Implementation
gym.register_envs(ale_py)      # Registering the ALE namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import cv2

# --------------------------------------------------
# General Tools/Weight Initialization
# --------------------------------------------------

def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --------------------------------------------------
# Image preprocessing (shared by Task2/Task3)
# --------------------------------------------------
class LazyFrames:
    """Share memory between stacked frames to save RAM (Task 3)."""
    def __init__(self, frames: List[np.ndarray]):
        self._frames = list(frames)
        self._out = None

    def __array__(self, dtype=None, copy=False):
        if self._out is None:
            self._out = np.stack(self._frames, axis=0)
            self._frames = None  # drop reference
        arr = self._out
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr.copy() if copy else arr

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, idx):
        return self.__array__()[idx]

class IdentityPreprocessor:
    def reset(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def step(self, obs):
        return np.asarray(obs, dtype=np.float32)

class AtariPreprocessor:
    """Grayscale → Resize(84×84) → stack 4 Frames"""
    def __init__(self, frame_stack: int = 4, lazy: bool = False):
        self.frames: Deque[np.ndarray] = deque(maxlen=frame_stack)
        self.frame_stack = frame_stack
        self.use_lazy = lazy

    def _proc(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        f = self._proc(obs)
        self.frames = deque([f] * self.frame_stack, maxlen=self.frame_stack)
        return LazyFrames(self.frames) if self.use_lazy else np.stack(self.frames, 0)

    def step(self, obs):
        self.frames.append(self._proc(obs))
        return LazyFrames(self.frames) if self.use_lazy else np.stack(self.frames, 0)

# --------------------------------------------------
# Network Architecture
# --------------------------------------------------
class MLP_DQN(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.network(x)

class CNN_DQN(nn.Module):
    """Nature DQN CNN (Task 2)"""
    def __init__(self, in_ch: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x / 255.0)

class DuelingDQN(nn.Module):
    """Dueling architecture with shared CNN feature extractor (Task 3)"""
    def __init__(self, in_ch: int, num_actions: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_val = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, 1))
        self.fc_adv = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, x):
        x = x / 255.0
        f = self.feature(x)
        v = self.fc_val(f)
        a = self.fc_adv(f)
        return v + (a - a.mean(dim=1, keepdim=True))

# --------------------------------------------------
# Task 3 only: Determine the priority of ER and N-step buffers
# --------------------------------------------------
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class PERBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 1_000_000):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pos = 0
        self.size = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.buffer: List[Transition] = [None] * self.capacity
        self.frame = 1  # count how many samples drawn for beta annealing

    def push(self, transition: Transition):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        beta = min(1.0, self.beta_start + (self.frame / self.beta_frames) * (1.0 - self.beta_start))
        self.frame += 1
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return self.size

class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque(maxlen=n)

    def push(self, transition: Transition):
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None
        R, next_state, done = 0.0, self.buffer[-1].next_state, self.buffer[-1].done
        for idx, tr in enumerate(self.buffer):
            R += (self.gamma ** idx) * tr.reward
            if tr.done:
                break
        state, action = self.buffer[0].state, self.buffer[0].action
        return Transition(state, action, R, next_state, done)

    def reset(self):
        self.buffer.clear()

# --------------------------------------------------
# Task 1 & Task 2：basic DQN Agent
# --------------------------------------------------
class BasicDQNAgent:
    """Vanilla DQN (MLP for vector obs, Nature CNN for image obs). 支援 CartPole 與 Pong baseline"""

    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env_name, render_mode=args.render_mode)
        self.test_env = gym.make(args.env_name, render_mode="rgb_array")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = self.env.action_space.n

        obs_shape = self.env.observation_space.shape
        if len(obs_shape) == 1:  # Task 1
            self.task_num = 1
            self.processor = IdentityPreprocessor()
            self.q_net = MLP_DQN(obs_shape[0], self.num_actions).to(self.device)
            self.target_net = MLP_DQN(obs_shape[0], self.num_actions).to(self.device)
        else:  # Task 2
            self.task_num = 2
            self.processor = AtariPreprocessor(lazy=False)
            self.q_net = CNN_DQN(4, self.num_actions).to(self.device)
            self.target_net = CNN_DQN(4, self.num_actions).to(self.device)

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.memory: Deque[Tuple] = deque(maxlen=args.memory_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.gamma = args.gamma
        self.eps, self.eps_decay, self.eps_min = args.eps_start, args.eps_decay, args.eps_min
        self.env_step = self.learn_step = 0
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------- replay memory helpers ----------
    def store(self, tr):
        self.memory.append(tr)

    def sample(self):
        batch = random.sample(self.memory, self.args.batch_size)
        return map(np.array, zip(*batch))

    # ---------- epsilon‑greedy ----------
    def select_action(self, s):
        if random.random() < self.eps:
            return self.env.action_space.sample()
        s_t = torch.from_numpy(np.array(s)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(s_t).argmax().item()

    # ---------- training step ----------
    def train_step(self):
        if len(self.memory) < self.args.replay_start:
            return
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        s, a, r, s2, d = self.sample()
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(self.device)
        a = torch.tensor(a).unsqueeze(1).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)
        q = self.q_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
        target = r + (1 - d) * self.gamma * q_next
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.args.target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        if self.learn_step % 1000 == 0:
            wandb.log({"loss": loss.item(), "env_step": self.env_step})

    # ---------- main loop ----------
    def run(self):
        proj = "DLP-Lab5-DQN-Pong" if self.task_num == 2 else "DLP-Lab5-DQN-CartPole"
        wandb.init(project=proj, name=self.args.run_name)
        ep = 0
        self.success_cnt = 0
        obs, _ = self.env.reset()
        state = self.processor.reset(obs)
        while ep < self.args.episodes:
            done, total, step = False, 0.0, 0
            while not done and step < self.args.max_steps:
                a = self.select_action(state)
                obs, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                nstate = self.processor.step(obs)
                self.store((state, a, r, nstate, float(done)))
                self.train_step()
                state = nstate
                total += r
                step += 1
                self.env_step += 1
                # Task 2 snapshots
                if self.task_num == 2 and self.env_step % 200_000 == 0:
                    torch.save(self.q_net.state_dict(), f"{self.save_dir}/snapshot_{self.env_step//1000}k_task2.pt")
            wandb.log({"episode": ep, "reward": total, "epsilon": self.eps, "env_step": self.env_step})
            print(f"Ep {ep:4d} | r={total:5.1f} | eps={self.eps:.3f} | step={self.env_step}")
            
            # Task1 save and early-stop
            if self.task_num==1 and total>=500:
                avg=self.evaluate(20)
                if avg>=0:
                    torch.save(self.q_net.state_dict(),os.path.join(self.save_dir,f"best_model_task1.pt"))
                    print(f"  [Save] best_model_task1.pt" )
                    print("Solved CartPole, stop.")
                    break
                    
            # Task2 snapshot & early-stop
            if self.task_num==2 and total >= 17:
                pth=os.path.join(self.save_dir,f"model_task2_ep{ep}_r{int(total)}.pt")
                torch.save(self.q_net.state_dict(),pth)
                print(f"  [Snapshot] saved {pth}")
                mean=self.evaluate(20)
                if mean >= 19:
                    self.success_cnt += 1
                    print("Solved Pong reward over 19 scores.")
                else:
                    print(f"Average score only {mean:.1f} < 19, continue training…")
                
                if self.success_cnt >= 3:
                    print("Solved Pong after 3 consecutive successes, stop.")
                    break
            ep += 1
            obs, _ = self.env.reset()
            state = self.processor.reset(obs)
    
    def evaluate(self, n=20):
        # Switch to evaluation mode and turn off epsilon
        self.q_net.eval()
        orig_eps = self.eps
        self.eps = 0.0

        rs = []
        for _ in range(n):
            obs, _ = self.test_env.reset()
            s = self.processor.reset(obs)
            done, tot = False, 0.0

            while not done:
                with torch.no_grad():
                    a = self.q_net(
                        torch.from_numpy(np.array(s))
                            .float()
                            .unsqueeze(0)
                            .to(self.device)
                    ).argmax().item()
                obs, r, term, trunc, _ = self.test_env.step(a)
                done = term or trunc
                tot += r
                s = self.processor.step(obs)
            rs.append(tot)

        mean = np.mean(rs)
        print(f"Eval {n} → {mean:.1f}")

        # Restore training mode and original epsilon
        self.eps = orig_eps
        self.q_net.train()

        return mean

# --------------------------------------------------
# Task3: Enhanced DQN Agent
# --------------------------------------------------
class EnhancedDQNAgent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(args.env)
        self.test_env = gym.make(args.env, render_mode="rgb_array")
        self.processor = AtariPreprocessor(4, lazy=True)
        self.num_actions = self.env.action_space.n
        self.gamma = args.gamma
        self.n_step = args.n_step
        self.online = DuelingDQN(4, self.num_actions).to(self.device)
        self.target = DuelingDQN(4, self.num_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=args.lr)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.steps, eta_min=1e-6)
        self.replay = PERBuffer(args.capacity, alpha=args.per_alpha,
                                beta_start=args.per_beta_start,
                                beta_frames=args.steps)
        self.nbuffer = NStepBuffer(self.n_step, self.gamma)
        self.batch_size = args.batch_size
        self.learn_start = 20_000
        self.update_freq = args.target_update

    # ---------- helpers ----------
    def select_action(self, state, eps):
        if random.random() < eps:
            return self.env.action_space.sample()
        st = torch.from_numpy(np.array(state, dtype=np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.online(st).argmax(dim=1).item()

    def push_transition(self, tr: Transition):
        ntr = self.nbuffer.push(tr)
        if ntr is not None:
            self.replay.push(ntr)
        if tr.done:
            self.nbuffer.reset()

    def update(self):
        if len(self.replay) < max(self.batch_size, self.learn_start):
            return
        transitions, indices, weights = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        states = np.stack([np.array(f) for f in batch.state], axis=0)
        next_states = np.stack([np.array(f) for f in batch.next_state], axis=0)
        states = torch.from_numpy(states.astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(next_states.astype(np.float32)).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        weights = weights.to(self.device)
        q_vals = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q
        td_error = target - q_vals
        loss = (weights * td_error.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        new_prios = td_error.abs().detach().cpu().numpy() + 1e-6
        self.replay.update_priorities(indices, new_prios)

    def evaluate(self, episodes=20):
        total = 0.0
        for _ in range(episodes):
            obs, _ = self.test_env.reset()
            state = self.processor.reset(obs)
            done = False
            while not done:
                st = torch.from_numpy(np.array(state, dtype=np.float32)).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.online(st).argmax(dim=1).item()
                obs, reward, term, trunc, _ = self.test_env.step(action)
                reward = np.sign(reward)
                done = term or trunc
                total += reward
                state = self.processor.step(obs)
        return total / episodes

    # ---------- main train loop ----------
    def train(self):
        wandb.init(project="DLP-Lab5-task3", name=self.args.run_name, save_code=True)
        eps = self.args.eps_start
        obs, _ = self.env.reset()
        state = self.processor.reset(obs)
        total_reward, ep, best_eval = 0.0, 0, {}
        print(f"Device: {self.device}")
        for step in range(1, self.args.steps + 1):
            action = self.select_action(state, eps)
            obs, reward, term, trunc, _ = self.env.step(action)
            reward = np.sign(reward)
            done = term or trunc
            total_reward += reward
            next_state = self.processor.step(obs)
            self.push_transition(Transition(state, action, reward, next_state, done))
            state = next_state
            self.update()
            eps = max(self.args.eps_min, eps * self.args.eps_decay)
            if step % self.update_freq == 0:
                self.target.load_state_dict(self.online.state_dict())
            if done:
                ep += 1
                wandb.log({"episode": ep, "reward": total_reward, "epsilon": eps, "steps": step})
                print(f"Ep {ep:4d} | r={total_reward:5.1f} | eps={eps:.3f} | step={step}")
                if total_reward >= 16:
                    avg = self.evaluate(20)
                    interval = step // 200_000
                    if avg >= best_eval.get(interval, -float('inf')):
                        best_eval[interval] = avg
                        fname = os.path.join(self.args.snapshot_dir, f"best_{interval*200}k.pt")
                        torch.save(self.online.state_dict(), fname)
                        print(f"[Eval] avg {avg:.2f} > best, saved {fname}")

                total_reward = 0.0
                obs, _ = self.env.reset()
                state = self.processor.reset(obs)
            # regular snapshot
            if step in self.args.snapshots:
                fname = os.path.join(self.args.snapshot_dir, f"model_{step//1000}k.pt")
                torch.save(self.online.state_dict(), fname)
                print(f"[Snapshot] saved {fname}")
        print("Training complete!")

# --------------------------------------------------
# CLI & Entry point
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True, help="Choose task 1/2/3")

    # ------------------- Task 1 & 2 -------------------
    t12 = subparsers.add_parser("1", help="Task 1 (CartPole) / Task 2 (Pong Baseline)")
    t12.add_argument("--env-name", type=str, default="CartPole-v1")
    t12.add_argument("--episodes", type=int, default=800)
    t12.add_argument("--batch-size", type=int, default=32)
    t12.add_argument("--memory-size", type=int, default=60_000)
    t12.add_argument("--lr", type=float, default=1e-4)
    t12.add_argument("--gamma", type=float, default=0.99)
    t12.add_argument("--eps-start", type=float, default=1.0)
    t12.add_argument("--eps-decay", type=float, default=0.9995)
    t12.add_argument("--eps-min", type=float, default=0.05)
    t12.add_argument("--replay-start", type=int, default=50_000)
    t12.add_argument("--target-freq", type=int, default=10_000)
    t12.add_argument("--max-steps", type=int, default=20_000)
    t12.add_argument("--save-dir", type=str, default="./results")
    t12.add_argument("--run-name", type=str, default="run_t12")
    t12.add_argument("--render-mode", type=str, default=None)

    # ------------------- Task 3 -------------------
    t3 = subparsers.add_parser("3", help="Task 3 (Enhanced Pong DQN)")
    t3.add_argument('--env', type=str, default='ALE/Pong-v5')
    t3.add_argument('--lr', type=float, default=3.75e-5)
    t3.add_argument('--gamma', type=float, default=0.99)
    t3.add_argument('--n-step', type=int, default=5)
    t3.add_argument('--capacity', type=float, default=1e6)
    t3.add_argument('--batch-size', type=int, default=32)
    t3.add_argument('--per-alpha', type=float, default=0.6)
    t3.add_argument('--per-beta-start', type=float, default=0.4)
    t3.add_argument('--eps-start', type=float, default=1.0)
    t3.add_argument('--eps-decay', type=float, default=0.9999925)
    t3.add_argument('--eps-min', type=float, default=0.05)
    t3.add_argument('--target-update', type=int, default=8000)
    t3.add_argument('--steps', type=int, default=1_000_000)
    t3.add_argument('--snapshot-dir', type=str, default='./task3_snapshots')
    t3.add_argument('--snapshots', nargs='+', type=int, default=[200_000, 400_000, 600_000, 800_000, 1_000_000])
    t3.add_argument('--seed', type=int, default=0)
    t3.add_argument('--run-name', type=str, default='run_t3')

    args = parser.parse_args()
    set_seed(0)

    if args.task in {'1', '2'}:
        agent = BasicDQNAgent(args)
        agent.run()
    elif args.task == '3':
        os.makedirs(args.snapshot_dir, exist_ok=True)
        agent = EnhancedDQNAgent(args)
        agent.train()

if __name__ == "__main__":
    main()

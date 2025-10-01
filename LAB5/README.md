# Lab 5 — Value-Based Reinforcement Learning  
Deep Learning @ NYCU (Spring 2025, TAICA)

This project implements and compares **vanilla DQN** and several **enhanced DQN variants** on classic control and Atari environments. The goal is to study how Double DQN, Prioritized Experience Replay (PER), Multi-Step Return, and Dueling Networks affect **sample efficiency** and **final performance**.

---

## 📌 Overview
- **Task 1**: Vanilla DQN on **CartPole-v1**  
- **Task 2**: Vanilla DQN on **Pong-v5** (visual input)  
- **Task 3**: Enhanced DQN on **Pong-v5**  
  - Double DQN  
  - PER + Multi-step return (n=5)  
  - Dueling DQN architecture  
  - Cosine learning rate scheduling  

---

## ⚙️ Environment

| Package       | Version   |
|---------------|-----------|
| Python        | ≥ 3.8     |
| torch         | ≥ 2.0.0   |
| gymnasium     | 1.1.1     |
| ale-py        | ≥ 0.10.0  |
| opencv-python | latest    |
| wandb         | latest    |

---

## 📂 File Structure

```
LAB5_11028141_POJUI/
├─ report.pdf # Technical report
├─ demo.mp4 # 5–6 min demo video
├─ code/
│ ├─ dqn.py # Main implementation
│ ├─ task1_cartpole.py # Training CartPole
│ ├─ task2_pong.py # Training Pong
│ ├─ task3_enhanced.py # Enhanced DQN
│ ├─ test_model_task1.py # Evaluation script (CartPole)
│ ├─ test_model_task2.py # Evaluation script (Pong vanilla)
│ └─ test_model_task3.py # Evaluation script (Pong enhanced)
├─ task1_cartpole.pt # Best CartPole model
├─ task2_pong.pt # Best Pong vanilla model
├─ task3_pong200k.pt # Enhanced DQN snapshot @200k
├─ task3_pong400k.pt
├─ task3_pong600k.pt
├─ task3_pong800k.pt
└─ task3_pong1M.pt
```


---

## 🚀 How to Run

### Task 1 — CartPole (Vanilla DQN)
```bash
python code/task1_cartpole.py --episodes 3000
python code/test_model_task1.py --model task1_cartpole.pt --eps 20 --render
```


### Task 2 — Pong (Vanilla DQN)
```bash
python code/task2_pong.py --steps 6500000
python code/test_model_task2.py --model task2_pong.pt --episodes 10 --output eval_videos
```


### Task 3 — Pong (Enhanced DQN)
```bash
python code/task3_enhanced.py --steps 1000000
python code/test_model_task3.py --model task3_pong800k.pt --episodes 10 --output eval_videos_task3
```

---

## 📊 Results

- **Task 1 (CartPole-v1)**: solved in ~30k steps, stable avg reward = 500  
- **Task 2 (Pong-v5, Vanilla)**: required ~6.5M steps to reach avg score 19  
- **Task 3 (Pong-v5, Enhanced)**: reached avg score 19 within ~735k steps (~8× faster)  

### Sample Efficiency Comparison

| Method                  | Steps to ≥19 | Relative |
|--------------------------|--------------|----------|
| Vanilla DQN (Task 2)    | ~6.5M        | 1×       |
| Enhanced DQN (Task 3)   | ~0.735M      | ↓ ~89%   |

---

## 🎥 Demo  
- [Demo Video (5–6 min)](./LAB5.mp4)  
  Includes code walkthrough and evaluation runs for Tasks 1–3.  

---

## 📑 References
- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015  
- Hessel et al., *Rainbow: Combining improvements in deep RL*, AAAI 2018  
- Schaul et al., *Prioritized Experience Replay*, ICLR 2016  


#!/usr/bin/env python3
import numpy as np

from roadsound_env import RoadSoundEnv, load_energy_bins  # 경로 맞게

N_STATES = 5
N_ACTIONS = 2

ALPHA = 0.1
GAMMA = 0.95
N_EPISODES = 200
EPSILON_START = 1.0
EPSILON_END = 0.1

def epsilon_by_episode(ep: int) -> float:
    frac = ep / max(1, N_EPISODES - 1)
    return float(EPSILON_START + (EPSILON_END - EPSILON_START) * frac)

def train():
    train_bins = load_energy_bins("train_bins.npy")
    env = RoadSoundEnv(train_bins)

    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    episode_returns = []

    rng = np.random.default_rng(42)

    for ep in range(N_EPISODES):
        s = env.reset()
        done = False
        total_r = 0.0
        eps = epsilon_by_episode(ep)

        while not done:
            # epsilon-greedy
            if rng.random() < eps:
                a = rng.integers(0, N_ACTIONS)
            else:
                a = int(np.argmax(Q[s]))

            next_state, r, done, info = env.step(a)
            total_r += r

            if not done:
                td_target = r + GAMMA * float(np.max(Q[next_state]))
            else:
                td_target = r

            td_error = td_target - Q[s, a]
            Q[s, a] += ALPHA * td_error

            if not done:
                s = next_state

        episode_returns.append(total_r)

        if (ep + 1) % 10 == 0:
            last10 = episode_returns[-10:]
            print(f"[train] ep={ep+1}/{N_EPISODES}, eps={eps:.3f}, "
                  f"mean_return(last10)={np.mean(last10):.3f}")

    np.save("q_table.npy", Q)
    np.save("qlearning_train_returns.npy", np.array(episode_returns, dtype=np.float32))
    print("[train] 학습 완료, q_table.npy / qlearning_train_returns.npy 저장")

if __name__ == "__main__":
    train()


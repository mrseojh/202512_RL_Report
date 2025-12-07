#!/usr/bin/env python3
import json
import numpy as np
from typing import Optional

from roadsound_env import RoadSoundEnv, load_energy_bins

def rule_policy(s: int) -> int:
    return 1 if s >= 3 else 0

def q_policy(s: int, Q: np.ndarray) -> int:
    return int(np.argmax(Q[s]))

def evaluate_policy(env: RoadSoundEnv, policy_fn, Q: Optional[np.ndarray] = None) -> float:
    s = env.reset()
    done = False
    total_r = 0.0
    while not done:
        if Q is None:
            a = policy_fn(s)
        else:
            a = policy_fn(s, Q)
        s_next, r, done, info = env.step(a)
        total_r += r
        if not done:
            s = s_next
    return total_r

def main():
    val_bins = load_energy_bins("val_bins.npy")
    env_rule = RoadSoundEnv(val_bins)
    env_q = RoadSoundEnv(val_bins)

    Q = np.load("q_table.npy")

    rule_total = evaluate_policy(env_rule, rule_policy)
    q_total = evaluate_policy(env_q, q_policy, Q=Q)

    print(f"[eval] Rule-based total reward : {rule_total:.3f}")
    print(f"[eval] Q-learning total reward: {q_total:.3f}")

    result = {
        "rule_total_reward": float(rule_total),
        "qlearning_total_reward": float(q_total),
    }
    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[eval] comparison_results.json 저장 완료")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_training_returns():
    returns = np.load("qlearning_train_returns.npy")
    episodes = np.arange(1, len(returns) + 1)

    plt.figure()
    plt.plot(episodes, returns, label="Return")

    # moving average 10
    window = 10
    if len(returns) >= window:
        cumsum = np.cumsum(np.insert(returns, 0, 0))
        ma = (cumsum[window:] - cumsum[:-window]) / window
        plt.plot(episodes[window-1:], ma, label=f"Moving Avg ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_returns.png")
    print("[plot] training_returns.png 저장")

def plot_policy_comparison():
    with open("comparison_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    rule_r = res["rule_total_reward"]
    q_r = res["qlearning_total_reward"]

    plt.figure()
    labels = ["Rule-based", "Q-learning"]
    vals = [rule_r, q_r]
    plt.bar(labels, vals)
    plt.ylabel("Total Reward")
    plt.title("Policy Comparison on Validation Data")
    plt.tight_layout()
    plt.savefig("policy_comparison.png")
    print("[plot] policy_comparison.png 저장")

def main():
    plot_training_returns()
    plot_policy_comparison()

if __name__ == "__main__":
    main()


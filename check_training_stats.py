# check_training_stats.py
import numpy as np

returns = np.load("qlearning_train_returns.npy")
print("전체 에피소드 수:", len(returns))

first_n = 20
last_n = 20

first = returns[:first_n]
last = returns[-last_n:]

print(f"초반 {first_n}개 에피소드 평균:", np.mean(first))
print(f"후반 {last_n}개 에피소드 평균:", np.mean(last))

print("초반 일부:", first[:5])
print("후반 일부:", last[-5:])


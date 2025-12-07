# check_env_debug.py
import numpy as np
from roadsound_env import RoadSoundEnv, load_energy_bins

bins = load_energy_bins("train_bins.npy")
env = RoadSoundEnv(bins)

s = env.reset()
print("초기 상태 s0 =", s)

for t in range(10):
    # 이벤트인지 확인
    event = 1 if s >= 3 else 0
    # 테스트용: 이벤트면 Record, 아니면 Skip
    a = 1 if event == 1 else 0

    s_next, r, done, info = env.step(a)
    print(f"t={t}, s={s}, event={event}, a={a}, r={r}, next={s_next}, done={done}")

    if done:
        break
    s = s_next


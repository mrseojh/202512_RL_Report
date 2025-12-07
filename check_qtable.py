import numpy as np

Q = np.load("q_table.npy")
print("Q-table:\n", Q)

for s in range(5):
    best_a = int(np.argmax(Q[s]))
    print(f"state {s}: best action = {best_a} ({'Record' if best_a==1 else 'Skip'})")


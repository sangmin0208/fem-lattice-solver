# benchmark_dot.py
import numpy as np
import time
import os

os.environ["OMP_NUM_THREADS"] = "48"
os.environ["MKL_NUM_THREADS"] = "48"

N = 10000
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# Warm-up
np.dot(A, B)

# Measure
t0 = time.time()
C = np.dot(A, B)
t1 = time.time()

print(f"[np.dot] Time: {t1 - t0:.4f} sec")

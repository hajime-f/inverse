import numpy as np
import cupy as cp
import time

N = [1000, 5000, 8000, 10000]

for n in N:

    start = time.time()
    W_cpu = 2 * np.random.rand(n, n) -1
    inv_W_cpu = np.linalg.inv(W_cpu)
    
    print ("CPU: elapsed_time:{0}".format(time.time() - start) + "[sec]")
    
    start = time.time()
    W_gpu = 2 * cp.random.rand(n, n) -1
    inv_W_gpu = cp.linalg.inv(W_gpu)
    
    print ("GPU: elapsed_time:{0}".format(time.time() - start) + "[sec]")

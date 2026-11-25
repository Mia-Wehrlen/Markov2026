import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

## Q3.C 
def get_h1(N, R0_array): # helper to comp h1 vals for arraY of R0 vals , K=N/10
    K = int(N / 10)
    gamma = 1.0
    h1_list = []
    
    for R0 in R0_array:
        beta = R0 * gamma / N
        rho = [1.0]
        # Generate the rho factors iteratively to prevent underflow/overflow
        for k in range(1, K):
            pk = beta * k * (N - k)
            qk = gamma * k
            rho.append(rho[-1] * qk / pk)
        
        # h1 = \rho_0 / \sum_{j=0}^{K-1} \rho_j  where \rho_0 = 1
        h1 = 1.0 / sum(rho)
        h1_list.append(h1)
        
    return np.array(h1_list)

# gen R0 array for plotting
R0_vals = np.linspace(0.5, 3.0, 500)

# calc h1 curves
h1_N100 = get_h1(100, R0_vals)
h1_N1000 = get_h1(1000, R0_vals)

# plot!!
plt.figure(figsize=(10, 6))
plt.plot(R0_vals, h1_N100, label='act h_1 (N=100)')
plt.plot(R0_vals, h1_N1000, label='act h_1 (N=1000)')
plt.legend()
plt.title('Outbreak Prob at various R0')
plt.xlabel('R_0')
plt.ylabel('h_1 (Prob of hitting K = N/10)')
plt.show()


## Q3.D
def En(ab, B):# does this count as one line? -- why does this want to be one line?
    e_steps = solve_banded((1, 1), ab, B) 
    e1_dt = e_steps[0]
    return e1_dt
    
def sim_En(N=100, K=10, gamma=1.0, dt=0.005, R0=1.5, trials=50000):
    beta = R0 * gamma / N
    total_steps = 0
    
    for _ in range(trials):
        n = 1  #x0 = 1
        while 0 < n < K:
            pn = beta * n * (N - n) * dt
            qn = gamma * n * dt
            
            rand = np.random.rand()
            if rand < pn:
                n += 1
            elif rand < pn + qn:
                n -= 1
            total_steps += 1
            
    simulated_e1_dt = (total_steps / trials) * dt
    return simulated_e1_dt

# tabulate...
target_R0_vals = [0.5, 0.8, 1.5, 3.0]
gamma = 1
K = 10
N = 100 
for r0 in target_R0_vals:
    beta = r0 * gamma / N
    size = K - 1  #  n = 1, 2, ..., K-1
    ab = np.zeros((3, size))
    B = -np.ones(size)  # rhs vector of -1s
    
    for i in range(size): # set up the matrix... 
        n = i + 1
        pn = beta * n * (N - n)  # pn / dt
        qn = gamma * n           # qn / dt
        
        ab[1, i] = -(pn + qn) # main
        if i > 0:
            ab[2, i-1] = qn # lower
        if i < size - 1:
            ab[0, i+1] = pn # upper

    exact_time = En(ab,B)
    print(f"R0 = {r0:<3} | Expected Real Time (e1*dt): {exact_time:.4f}")

sim_val = sim_En(R0=3)
print(f"Simulated e1*dt (R0 = 3) : {sim_val:.4f} (exact value {exact_time:.4f})")

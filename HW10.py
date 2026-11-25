import numpy as np
from scipy.stats import poisson

# Parameters
lam = 12.0   
mu = 3.0     
rho = lam / mu
T = 500.0    

np.random.seed(42)
t, n = 0.0, 0
time_in_state = {}

while t < T:
    total_rate = lam + n * mu
    dt = np.random.exponential(1.0 / total_rate)
    if t + dt > T:
        time_in_state[n] = time_in_state.get(n, 0.0) + (T - t)
        break
    time_in_state[n] = time_in_state.get(n, 0.0) + dt
    t += dt
    n += 1 if np.random.rand() < lam / total_rate else -1

states = range(max(time_in_state.keys()) + 1)
emp_probs = [time_in_state.get(k, 0.0) / T for k in states]
theo_probs = [poisson.pmf(k, rho) for k in states]

diffs = [abs(e - th) for e, th in zip(emp_probs, theo_probs)]
max_diff = max(diffs)
time_avg = sum(k * p for k, p in zip(states, emp_probs))

# Smallest fleet size c with P(X > c) <= 0.01
c = next(k for k in range(50) if 1.0 - poisson.cdf(k, rho) <= 0.01)

print(f"Max diference: {max_diff:.4f} at state {np.argmax(diffs)}")
print(f"Time avg: {time_avg:.4f} vs rho = {rho}")
print(f"Smallest fleet size c: {c} (P(X > {c}) = {1 - poisson.cdf(c, rho):.4f})")

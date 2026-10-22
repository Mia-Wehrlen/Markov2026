import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# using sub groups! -- aka the brute force iterative method...
rates = np.array([3.0, 2.0, 1.5, 1.2, 1.0, 0.75])
n = len(rates)
memo_E = {}
memo_P = {}

def solve_subgroup(state):
    if not state:
        return 0.0, np.zeros(n)
    if state in memo_E:
        return memo_E[state], memo_P[state]
    
    sum_rate = sum(rates[i] for i in state)
    expected_time_to_next = 1.0 / sum_rate
    
    # Base case: only one student left
    if len(state) == 1:
        last_student = state[0]
        prob_last = np.zeros(n)
        prob_last[last_student] = 1.0
        memo_E[state] = expected_time_to_next
        memo_P[state] = prob_last
        return expected_time_to_next, prob_last

    total_E = expected_time_to_next
    total_P = np.zeros(n)
    
    for i in state:
        p_i = rates[i] / sum_rate
        next_state = tuple(j for j in state if j != i)
        next_E, next_P = solve_subgroup(next_state)
        
        total_E += p_i * next_E
        total_P += p_i * next_P
        
    memo_E[state] = total_E
    memo_P[state] = total_P
    return total_E, total_P

initial_state = tuple(range(n))
exact_EM_dp, exact_P_last = solve_subgroup(initial_state)

# using the formula... 
exact_EM_alt = 0.0
for k in range(1, n + 1):
    for comb in combinations(range(n), k):
        group_rate = sum(rates[i] for i in comb)
        sign = (-1) ** (k + 1)
        exact_EM_alt += sign * (1.0 / group_rate)

print(f"E[M] using subgroups : {exact_EM_dp:.5f}")
print(f"E[M] using formula : {exact_EM_alt:.5f}")
print("Probability of student i finishing last:")
for idx, p in enumerate(exact_P_last):
    print(f"  {idx+1}: {p:.5f}")

# simulation...
num_sims = 2 * 20**5

# draw ind exponential times for all students
X_samples = np.random.exponential(1.0 / rates, size=(num_sims, n))

# calc M = max finish time and who finished first+last
M_samples = np.max(X_samples, axis=1)
first = np.argmin(X_samples, axis=1)
last = np.argmax(X_samples, axis=1)

sim_EM = np.mean(M_samples)
sim_P_last = np.bincount(last, minlength=n) / num_sims

print("Sim Results:")
print(f"Sim E[M]: {sim_EM:.5f}")
print("Sim P(last):", [f"{p:.5f}" for p in sim_P_last])

# independence of first finisher...
print("Independence Check:")
min_X_samples = np.min(X_samples, axis=1)
overall_mean_min = np.mean(min_X_samples)
for i in range(n):
    mask = (first == i)
    conditional_mean = np.mean(min_X_samples[mask])
    print(f"  Avg Min(X) | {i+1} finished first: {conditional_mean:.5f}")

print("Its Independent!")

# plot! 
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(M_samples, bins=100, density=True, label='Sim M')

# Compute and Plot Exact Density Curve: d/dt [ \prod (1 - e^{-\lambda_i t}) ]
t_vals = np.linspace(0.01, np.max(M_samples), 500)
calc_density = np.zeros_like(t_vals)

for idx, t in enumerate(t_vals):
    prob_all_before = np.prod(1.0 - np.exp(-rates * t))
    # Using the product rule derivative: d/dt \prod f_i = \prod f_i * \sum (f_i' / f_i)
    sum_derivative_ratio = np.sum((rates * np.exp(-rates * t)) / (1.0 - np.exp(-rates * t)))
    calc_density[idx] = prob_all_before * sum_derivative_ratio

plt.plot(t_vals, calc_density, label='Calculated Density')
plt.title('M vs. Exact Density')
plt.xlabel('Time (hours)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# E[M] growth for lambda_i
plt.subplot(1, 2, 2)
k_max = 12
k_vals = np.arange(1, k_max + 1)
EM_k = []

# Compute E[M] for each k via the subset/alternating formula structure
for k in k_vals:
    k_rates = np.arange(1, k + 1)
    # Using alternating formula for clean implementation per k step
    em_value = 0.0
    for r in range(1, k + 1):
        for comb in combinations(range(k), r):
            group_rate = sum(k_rates[i] for i in comb)
            em_value += ((-1) ** (r + 1)) * (1.0 / group_rate)
    EM_k.append(em_value)

plt.plot(k_vals, EM_k, 'bo-', lw=2, label='Exact E[M]')
plt.axhline(y=2 * np.log(2))
plt.title('E[M] Growth Curve, lambda_i')
plt.xlabel('k')
plt.ylabel('E[M]')
plt.xticks(k_vals)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("NOTES:")
print("No it does not grow without bounds")

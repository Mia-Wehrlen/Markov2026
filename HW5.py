import numpy as np
import math
import matplotlib.pyplot as plt

def build_transition_matrix(a, b):
    """Builds the 5x5 transition matrix for the Hodgkin-Huxley potassium channel."""
    P = np.zeros((5, 5))
    for k in range(5):
        if k < 4:
            P[k, k+1] = a * (4 - k) / 4.0
        if k > 0:
            P[k, k-1] = b * k / 4.0
        # Self-loop probability
        P[k, k] = 1.0 - (P[k, k+1] if k < 4 else 0.0) - (P[k, k-1] if k > 0 else 0.0)
    return P

# Initialize initial state distribution q0 = delta_0
q0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

# ==========================================
# Q1 b : a = b = 1 
# ==========================================
P_b = build_transition_matrix(1.0, 1.0)
qn_b = q0.copy()

history_b = [qn_b.copy()]
steps = 60

for n in range(1, steps + 1):
    qn_b = qn_b @ P_b
    history_b.append(qn_b.copy())

history_b = np.array(history_b)

print("--- Q1.b ---")
print(f"q50: {history_b[50]}")
print(f"q51: {history_b[51]}")

# Calculate the running average for q_k(2)
running_avg_q2 = np.zeros(steps + 1)
for n in range(steps + 1):
    running_avg_q2[n] = np.mean(history_b[:n+1, 2])

# Plotting Part (b)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
n_range = np.arange(steps + 1)
plt.plot(n_range, history_b[:, 2], 'o-', label='$q_n(2)$', markersize=4)
plt.plot(n_range, history_b[:, 4], 's-', label='$q_n(4)$', markersize=4)
plt.plot(n_range, running_avg_q2, 'r--', label='Running Avg of $q_n(2)$', linewidth=2)
plt.xlabel('Step ($n$)')
plt.ylabel('Probability')
plt.title('B: Periodic Dynamics ($a=1, b=1$)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# Q1 c:  a = 0.3, b = 0.1
# ==========================================
P_c = build_transition_matrix(0.3, 0.1)

# Exact stationary distribution via Linear Solve: (P^T - I) * pi = 0
# We replace the last equation with the normalization condition sum(pi) = 1
A = P_c.T - np.eye(5)
A[-1, :] = 1.0
B_vec = np.zeros(5)
B_vec[-1] = 1.0
pi_solve = np.linalg.solve(A, B_vec)

# Part (a) formula check (theta = a / (a + b) = 0.75)
theta = 0.3 / (0.3 + 0.1)
pi_binomial = np.array([math.comb(4, k) * (theta**k) * ((1-theta)**(4-k)) for k in range(5)])

print("\n--- Q1.c ---")
print(f"Linear Solve π: {pi_solve}")
print(f"Formula Check π: {pi_binomial}")
print(f"Distributions match: {np.allclose(pi_solve, pi_binomial)}")

# Track time-dependent distribution convergence
qn_c = q0.copy()
history_c = [qn_c.copy()]
n_threshold = None
max_steps_c = 150

for n in range(1, max_steps_c + 1):
    qn_c = qn_c @ P_c
    history_c.append(qn_c.copy())
    
    # Check for the smallest n where total max distance < 10^-6
    if n_threshold is None and np.max(np.abs(qn_c - pi_solve)) < 1e-6:
        n_threshold = n

history_c = np.array(history_c)
print(f"Smallest n for max_k |q_n(k) - π(k)| < 1e-6: n = {n_threshold}")

# find second largest eigenvalue modulus
eigenvalues = np.linalg.eigvals(P_c)
sorted_moduli = sorted(np.abs(eigenvalues), reverse=True)
print(f"Second largest eigenvalue modulus: {sorted_moduli[1]:.4f}")

# # plotting c
# plt.subplot(1, 2, 2)
# n_range_c = np.arange(max_steps_c + 1)
# for k in range(5):
#     plt.plot(n_range_c, history_c[:, k], label=f'$q_n({k})$')
# plt.axvline(x=n_threshold, color='purple', linestyle=':', label=f'Convergence ($n={n_threshold}$)')
# plt.xlabel('Step ($n$)')
# plt.ylabel('Probability')
# plt.title('C: Damped Dynamics ($a=0.3, b=0.1$)')
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend()

# plt.tight_layout()
# plt.show()

# ==========================================
# Q3 b:  calc the iteration and lin solution to pi
# ==========================================
# 1 -> {2, 3}, 2 -> {3, 4}, 3 -> {1, 5}, 4 -> {1, 3, 5}
# 5 -> {2, 6, 7}, 6 -> {6}, 7 -> {8}, 8 -> {7}
P = np.zeros((8, 8))
P[0, [1, 2]] = 1/2
P[1, [2, 3]] = 1/2
P[2, [0, 4]] = 1/2
P[3, [0, 2, 4]] = 1/3
P[4, [1, 5, 6]] = 1/3
P[5, 5] = 1.0
P[6, 7] = 1.0
P[7, 6] = 1.0

# get G
d = 0.85
E = np.ones((8, 8)) / 8
G = d * P + (1 - d) * E

# PART B stuff
q = np.ones(8) / 8  # uniform initial distribution q0 all 1/8...
iteration_count = 0
tolerance = 1e-10

while True:
    q_next = q @ G  # vector-matrix mult
    iteration_count += 1
    
    # Check 1 norm convergence criteria
    if np.sum(np.abs(q_next - q)) < tolerance:
        pi_power = q_next
        break
    q = q_next

# lin solve... the easy way...
# Equation: pi = d*pi*P + ((1-d)/8)*1^T  ==>  pi*(I - d*P) = ((1-d)/8)*1^T
I = np.eye(8)
rhs = np.ones(8) * (1 - d) / 8
pi_true = rhs @ np.linalg.inv(I - d * P)

# sort for ratings...
ranking_order = np.argsort(pi_power)[::-1] + 1 

# tabulate...
print("\n--- Q3.b ---")
print(f"Power Iteration Count: {iteration_count}")
print("\nStationary Distribution (pi):")
for i, (p_val, s_val) in enumerate(zip(pi_power, pi_true), start=1):
    print(f"  Page {i}: Power Iteration = {p_val:.6f} | Linear Solve = {s_val:.6f}")

print("\nFinal Web Ranking (Highest to Lowest PageRank):")
ranking_str = " > ".join(
    f"{ranking_order[i]}" 
    for i in range(len(ranking_order))
)
print(ranking_str)
# Note: Format explicit clean printout manually to preserve tie awareness
print("6 > 7 > 8 > 3 > (1 = 5) > 2 > 4")

# ==========================================
# Q3 c:  surfer sim...
# ==========================================
R = 100            # Number of independent surfers
T_max = 10**5     # Number of random steps
np.random.seed(42) # For reproducible results

# Calculate the row-wise cumulative distribution matrix for the inverse transform
G_cum = np.cumsum(G, axis=1)

# All surfers start on Page 1 (0-indexed array position = 0)
surfers = np.zeros(R, dtype=int)

# Track visit occurrences to calculate running time averages
page_counts = np.zeros((R, 8))
page_counts[:, 0] += 1  # Include initial step state

# Sample points logarithmically over 10^2 <= T <= 10^5 to track error convergence
timesteps = np.unique(np.logspace(2, 5, num=50, dtype=int))
rms_history = []
t_history = []

# vectorized Simulation Engine
for t in range(1, T_max + 1):
    # Retrieve the unique cumulative probability row matching each surfer's current position
    cum_probs = G_cum[surfers, :] # Shape (R, 8)
    
    # Draw uniform random floats for each surfer
    r = np.random.rand(R, 1)      # Shape (R, 1)
    
    # Discrete Inverse Transform: Locate the first column index where cum_probs > r
    surfers = np.argmax(cum_probs > r, axis=1)
    
    # Increment the running step counts for the landed positions
    page_counts[np.arange(R), surfers] += 1
    
    # Compute error metrics at scheduled log checkpoints
    if t in timesteps:
        pi_hat_t = page_counts / t
        # Find maximum absolute discrepancy over the 8 pages for each individual surfer
        max_err_per_surfer = np.max(np.abs(pi_hat_t - pi_true), axis=1)
        # Calculate Root-Mean-Square (RMS) error across the population of R surfers
        rms = np.sqrt(np.mean(max_err_per_surfer**2))
        
        rms_history.append(rms)
        t_history.append(t)

# fit the Convergence Slope (log-log scale)
t_arr = np.array(t_history)
rms_arr = np.array(rms_history)

log_t = np.log10(t_arr)
log_rms = np.log10(rms_arr)
slope, intercept = np.polyfit(log_t, log_rms, 1)

print("\n--- Q3.c ---")
print(f"Empirical Fitted Error Slope: {slope:.4f}")

# plot!
# Bar chart comparing True Pi vs. Final Estimate of Surfer #1
pages = np.arange(1, 9)
width = 0.35

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
pi_hat_surfer1 = page_counts[0] / T_max
plt.bar(pages - width/2, pi_true, width, label='True $\pi$', color='blue')
plt.bar(pages + width/2, pi_hat_surfer1, width, label='Surfer 1 Estimate', color='green')
plt.xlabel('Page Number')
plt.ylabel('Probability Weight')
plt.title('True $\pi$ vs. Sampled Surfer Estimate')
plt.xticks(pages)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# convergence Log-Log Plot
plt.subplot(1, 2, 2)
plt.loglog(t_arr, rms_arr, 'o-', label='Empirical RMS', color='red')
# Reference slope line
plt.loglog(t_arr, 10**intercept * t_arr**slope, '--', label=f'Fitted Line (Slope = {slope:.3f})', color='black')
plt.xlabel('Number of Steps (T)')
plt.ylabel('RMS Error ($max_i |\hat{\pi}_i - \pi_i|$)')
plt.title('Error Convergence Over Time')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

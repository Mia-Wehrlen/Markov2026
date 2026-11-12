import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

Q = np.array([
    [-2.0,  2.0,  0.0,  0.0],
    [ 1.0, -3.0,  2.0,  0.0],
    [ 0.0,  2.0, -4.0,  2.0],
    [ 0.0,  0.0,  3.0, -3.0]
])

x0 = np.array([1.0, 0.0, 0.0, 0.0])  # Starting empty
pi = np.array([3/19, 6/19, 6/19, 4/19])  # Stationary distribution

# 2. Find eigenvalues and spectral gap lambda_2
eigvals = np.linalg.eigvals(Q)
eigvals_sorted = np.sort(eigvals.real)  # Sorted ascending: [lambda_4, lambda_3, lambda_2, 0]
lambda_2 = eigvals_sorted[-2]

def get_x(t):
    return x0 @ expm(t * Q)

def get_tv(t):
    return 0.5 * np.sum(np.abs(get_x(t) - pi))

# Compute values over t in [0, 5]
t_vals = np.linspace(0, 5, 500)
x_vals = np.array([get_x(t) for t in t_vals])
tv_vals = np.array([get_tv(t) for t in t_vals])

# calc exact time when TV(t) = 0.01 and compare with ln(100)/|lambda_2|
t_tv_01 = brentq(lambda t: get_tv(t) - 0.01, 0, 10)
t_approx = np.log(100) / abs(lambda_2)

# tabulate!!
print("--- Value Comparison ---")
print(f"Time for TV(t) = 0.01: {t_tv_01:.4f} hours")
print(f"Approx ln(100)/|l2|: {t_approx:.4f} hours")

# plot!
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# state probs... x_i(t)
for i in range(4):
    ax1.semilogy(t_vals, x_vals[:, i], label=f'x_{i}(t) (State {i})')

ax1.set_title('State Probabilities x_i(t)')
ax1.set_xlabel('Hours')
ax1.set_ylabel('Probability x_i(t)')
ax1.set_ylim(1e-3, 1.2)
ax1.grid(True, which="both")
ax1.legend()
# total Variation Distance TV(t) and reference slope lambda_2
ax2.semilogy(t_vals, tv_vals, label='TV(t)', color='black')
# Reference line with slope lambda_2 (anchored at TV(0) for visual comparison)
ref_line = tv_vals[0] * np.exp(lambda_2 * t_vals)
ax2.semilogy(t_vals, ref_line, '--', color='red', label=f'Slope lambda_2 = {lambda_2:.3f}$')
# Mark TV = 0.01 threshold
ax2.axhline(0.01, label='TV = 0.01')
ax2.plot(t_tv_01, 0.01, 'bo', label=f't_0.01 = {t_tv_01:.2f} hrs')
ax2.set_title('Total Variation Distance TV(t)')
ax2.set_xlabel('Hours')
ax2.set_ylabel('TV(t)')
ax2.set_ylim(1e-4, 1.5)
ax2.grid(True, which="both")
ax2.legend()

plt.tight_layout()
plt.show()

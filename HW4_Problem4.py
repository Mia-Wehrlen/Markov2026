#
# HW4_Problem4.py
#
# folded proteins... 
# 

import numpy as np
import matplotlib.pyplot as plt

# P matrix...
P = np.array([
    [0.00, 0.50, 0.50, 0.00, 0.00],  # U
    [0.25, 0.00, 0.00, 0.50, 0.25],  # I
    [0.75, 0.00, 0.00, 0.00, 0.25],  # M
    [0.00, 0.00, 0.00, 1.00, 0.00],  # F 
    [0.00, 0.00, 0.00, 0.00, 1.00]   # A 
])

np.random.seed(666)
rng = np.random.default_rng()
num_chains = 10**4
start_state = 1  # State I

folded_times = []
aggregated_times = []

results = {}

# get cumulative transition probs for inverse transform ...
P_cumulative = np.cumsum(P, axis=1)

# simulate - x = U, I, M
def simulate(start, n_chains=10**4):
    times, fates = [], []
    for _ in range(n_chains):
        x, t = start, 0
        while x not in (3, 4):
            u = rng.random()
            x = np.searchsorted(P_cumulative[x], u)
            t += 1
        times.append(t)
        fates.append(x)

        if start == 1: # collect if start is I for second part...
            if x == 3: # collect times to F for later...
                folded_times.append(t)
            elif x == 4:
                aggregated_times.append(t) # collect times to A for later...
    return np.array(times), np.array(fates)

results = {}
for start, name in [(0,'U'),(1,'I'),(2,'M')]:
    times, fates = simulate(start)
    fold = fates == 3
    results[name] = {
        'h_hat':   fold.mean(),
        'g_hat':   times.mean(),
        'tauF_hat': times[fold].mean() if fold.any() else np.nan,
        'tauA_hat': times[~fold].mean() if (~fold).any() else np.nan,
    }
    
# tabulate...

print(f"{'x':>4} {'h_hat':>8} {'g_hat':>8} {'tauF_hat':>10} {'tauA_hat':>10}")
exact = {'U':(0.5,4,4,4),'I':(5/8,2,9/5,7/3),'M':(3/8,4,5,17/5)}
for name in ['U','I','M']:
    r = results[name]
    e = exact[name]
    print(f"{name:>4} {r['h_hat']:>8.4f} {r['g_hat']:>8.4f} "
          f"{r['tauF_hat']:>10.4f} {r['tauA_hat']:>10.4f}")
    print(f"{'exact':>4} {e[0]:>8.4f} {e[1]:>8.4f} {e[2]:>10.4f} {e[3]:>10.4f}")

# exact pmfs...

# these are the relevant sub matricies.... 
Q = P[0:3, 0:3]  # transient block
R = P[0:3, 3:5]  # transient to recurrent block

# prob of getting absorbed from I into F and A...
P_absorb = (
    5/8, # F
    3/8  # A
)

max_n = 25  # space it out....
n_values = np.arange(1, max_n + 1)

PIA = []
PIF = []

# Q^0 is the Identity (I)
Q_pow = np.eye(3)

for n in n_values:
    # (Q^(n-1) * R)_{I, emp} / P_I(emp)
    # state I is index 1. empircaal F is column index 0 in R, empirical A is column index 1 in R.
    prob_F = (Q_pow @ R)[1, 0] / P_absorb[0]
    prob_A = (Q_pow @ R)[1, 1] / P_absorb[1]
    
    PIF.append(prob_F)
    PIA.append(prob_A)
    
    # Update Q power for next iteration
    Q_pow = Q_pow @ Q


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
# plot! -- fold...
axes[0].hist(folded_times, bins=np.arange(0.5, max_n + 1.5, 1), density=True, 
             alpha=0.6, label='Simulation')
axes[0].scatter(n_values, PIF, color='darkblue', zorder=3, label='Exact PMF')
axes[0].plot(n_values, PIF, color='darkblue', linestyle='--', alpha=0.5)
axes[0].set_title('Given Folding (F)')
axes[0].set_xlabel('Time Steps ($T = n$)')
axes[0].set_ylabel('Conditional Probability')
axes[0].set_xlim(0, max_n)
axes[0].grid(axis='y', alpha=0.3)
axes[0].legend()

# plot! -- agregate..
axes[1].hist(aggregated_times, bins=np.arange(0.5, max_n + 1.5, 1), density=True, 
             alpha=0.6, label='Simulation')
axes[1].scatter(n_values, PIA, color='darkred', zorder=3, label='Exact PMF')
axes[1].plot(n_values, PIA, color='darkred', linestyle='--', alpha=0.5)
axes[1].set_title('Given Aggregating (A)')
axes[1].set_xlabel('Time Steps ($T = n$)')
axes[1].set_xlim(0, max_n)
axes[1].grid(axis='y', alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

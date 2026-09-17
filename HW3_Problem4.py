#
# HW3_Problem4.py
#
# lion and lamb -- markov chains
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import linregress

rng = np.random.default_rng(42)

R = 20**4  # independent lambs
T = 10**4   # time steps
d0 = 10      # initial separation

# Part b -- N=1 (one lion and many lambs)
alive1 = np.ones(R, dtype=bool) # all lambs begin alive...
D1 = np.full(R, d0, dtype=np.int32)   # starting D between lambs and lion
S1 = np.zeros(T + 1) # empty survival array
S1[0] = 1 # all lambs begin alive!

for t in range(1, T + 1):
     # Lamb hops: +1 or -1
    lamb_step = rng.integers(0, 2, size=R) * 2 - 1   # ±1
    # Lion 1 hops ind
    lion_step = rng.integers(0, 2, size=R) * 2 - 1

    D1 += (lion_step - lamb_step) * alive1 # you must have been alive and now share a spot with the lion...
    alive1 &= (D1 != 0) # he died :(
    S1[t] = alive1.mean() # proportion of lambs still alive...

# Part c -- N=2 (two lions and many lambs)
alive2 = np.ones(R, dtype=bool) # all lambs begin alive
D2a = np.full(R, d0, dtype=np.int32)  # starting D between lion 1 and lambs
D2b = np.full(R, d0, dtype=np.int32)  # starting D between lion 2 and lambs
S2 = np.zeros(T + 1) # empty survival array
S2[0] = 1 # all lambs begin alive!

for t in range(1, T + 1):
    # Lamb hops: +1 or -1
    lamb_step = rng.integers(0, 2, size=R) * 2 - 1   # ±1
    # Lion 1 hops ind
    lion1_step = rng.integers(0, 2, size=R) * 2 - 1
    # Lion 2 hops ind
    lion2_step = rng.integers(0, 2, size=R) * 2 - 1

    D2a += (lion1_step - lamb_step) * alive2 # you must have been alive and now share a spot with the lion...
    D2b += (lion2_step - lamb_step) * alive2 # you must have been alive and now share a spot with the lion...
    alive2 &= (D2a != 0) & (D2b != 0) # he died :(
    S2[t] = alive2.mean()  # proportion of lambs still alive...

# Plot on log-log axes
t_vals = np.arange(1, T + 1)
S1_vals = S1[1:]
S2_vals = S2[1:]

# continum prediction...
t_cont = np.linspace(1, T, 5000)
S1_cont = erf(d0 / (2 * np.sqrt(t_cont)))

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(t_vals, S1_vals, color='blue', lw=0.8, label='Proportion of Lambs Alive (N=1)')
ax.loglog(t_vals, S2_vals, color='red', lw=0.8, label='Proportion of Lambs Alive (N=2)')
ax.loglog(t_cont, S1_cont, 'k--', lw=0.8, label='Continum Prediction')

# Fit slope over t in [10**2, 10**4]
mask = (t_vals >= 10**2) & (t_vals <= T)
log_t = np.log(t_vals[mask])
log_S = np.log(S1_vals[mask])
log_S2 = np.log(S2_vals[mask])

slope, intercept = np.polyfit(log_t, log_S, 1)
slope2, intercept2 = np.polyfit(log_t, log_S2, 1)

beta1 = -slope
beta2 = -slope2

print(f"Fitted slope (exponent beta_1): {beta1:.4f}")
print(f"Fitted slope (exponent beta_2): {beta2:.4f}")
print(f"Fit window: {10**2} to {10**4}")

fit_line = np.exp(intercept) * t_vals[mask] ** slope
ax.loglog(t_vals[mask], fit_line, 'r--', lw=1.5,
          label='Fit: slope (beta_1)')

fit_line2 = np.exp(intercept2) * t_vals[mask] ** slope2
ax.loglog(t_vals[mask], fit_line2, 'g--', lw=1.5,
          label='Fit: slope (beta_2)')

ax.set_xlabel('time (t)')
ax.set_ylabel('P(lamb is alive)')
ax.legend()
plt.tight_layout()
plt.savefig('survival_lambs.png', dpi=150)
plt.show()

# tabulate....
check_times = [10**2, 10**3, 10**4]
print(f"{'t':>6}  {'S2(t)':>10}  {'S1(t)^2':>10}")
for tc in check_times:
    print(f"{tc:>6}  {S2[tc]:>10.5f}  {S1[tc]**2:>10.5f}")

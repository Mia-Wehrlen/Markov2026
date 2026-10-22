import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# params!
lambda_1 = 4
hours = 200000
T = 1

np.random.seed(666)

# with exponential iinter-arrival gaps
gaps = np.random.exponential(1 / lambda_1, size=(hours, 25))
arrival_times_exp = np.cumsum(gaps, axis=1)

counts_exp = np.sum(arrival_times_exp <= T, axis=1)
t3_exp = arrival_times_exp[:, 2]  # Third meteor arrival times

# with poisson count + uniform points

counts_poiss = np.random.poisson(lambda_1 * T, size=hours)
arrival_times_poiss = []

for count in counts_poiss:
    if count == 0:
        arrival_times_poiss.append(np.array([]))
    else:
        arrival_times_poiss.append(np.sort(np.random.uniform(0, T, size=count)))


# plot!

fig, axes = plt.subplots(2, 2, figsize=(12, 12)) # many plots so 2x2 layout...

# PMF comparison hist
axes[0, 0].hist(counts_exp, bins=np.arange(15)-0.5, density=True, alpha=0.6, label='Simulated N(1)')
k_vals = np.arange(15)
axes[0, 0].plot(k_vals, stats.poisson.pmf(k_vals, 4), 'ro-', lw=2, label='Poisson(4) Dist')
axes[0, 0].set_title("N(1) Dist vs Poisson PMF")
axes[0, 0].legend()

# T3 density vs hist
axes[0, 1].hist(t3_exp, bins=50, density=True, alpha=0.6, label='Simulated T3')
t_vals = np.linspace(0, 3, 200)
axes[0, 1].plot(t_vals, stats.gamma.pdf(t_vals, a=3, scale=1/lambda_1), label='Gamma(3, 4) Dist Actual')
axes[0, 1].set_title("T3 Distribution vs Gamma Density")
axes[0, 1].set_xlabel("Time (hours)")
axes[0, 1].legend()

# raster -- first 10 
for i in range(10):
    times_exp = arrival_times_exp[i][arrival_times_exp[i] <= T]
    axes[1, 0].vlines(times_exp, i - 0.4, i + 0.4)
axes[1, 0].set_title("Raster using exp gaps")
axes[1, 0].set_xlabel("time (hours)")
axes[1, 0].set_ylabel("index")
axes[1, 0].set_ylim(-1, 10)

for i in range(10):
    times_poiss = arrival_times_poiss[i]
    axes[1, 1].vlines(times_poiss, i - 0.4, i + 0.4)
axes[1, 1].set_title("Raster using poiss and uniform points")
axes[1, 1].set_xlabel("time (hours)")
axes[1, 1].set_ylabel("index")
axes[1, 1].set_ylim(-1, 10)

plt.show()

mean_n1 = np.mean(counts_exp)
var_n1 = np.var(counts_exp)

print(f"Using Exponential Gaps: ")
print(f"Mean N(1): {mean_n1:.4f}")
print(f"Var N(1): {var_n1:.4f}\n")

# take N(1) = 3 ...
n3 = np.where(counts_exp == 3)[0]
counts_half_1 = np.sum(arrival_times_exp[n3] <= 0.5, axis=1)

p_half_zero = np.mean(counts_half_1 == 0)
e_t1_cond = np.mean(arrival_times_exp[n3, 0])

# Global correlation between first and second half hours
global_half_1 = np.sum(arrival_times_exp <= 0.5, axis=1)
global_half_2 = counts_exp - global_half_1
global_corr = np.corrcoef(global_half_1, global_half_2)[0, 1]

print(f"Given N(1) = 3: ")
print(f"Est P(N(1/2) = 0 | N(1) = 3): {p_half_zero:.4f} (Theoretical: 0.125)")
print(f"Est E[T1 | N(1) = 3]: {e_t1_cond:.4f} (Theoretical: 0.25)")
print(f"correlation between 1st & 2nd half-hour: {global_corr:.4f} (Theoretical: 0.0)\n")

# total variation distance slope
dts = np.array([1/10, 1/20, 1/50, 1/100, 1/200, 1/500, 1/1000, 1/2000])
tvs = []

for dt in dts:
    n = int(round(T / dt))
    p = lambda_1 * dt
    k = np.arange(0, 40)
    
    bin_pmf = stats.binom.pmf(k, n, p)
    bin_pmf[k > n] = 0
    poi_pmf = stats.poisson.pmf(k, lambda_1 * T)
    
    tv = 0.5 * np.sum(np.abs(bin_pmf - poi_pmf))
    tvs.append(tv)

slope, _ = np.polyfit(np.log(dts), np.log(tvs), 1)
print(f"Total Variation log-log Slope: {slope:.4f} (Theoretical: ~1.0)")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

np.random.seed(666)

scans = np.array([-1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97, 0.79, 1.50, 2.00, 2.63, 1.19, -2.31, 0.87])

# Q1.c

print("Q1.C : STEP SIZE VARIATION")

# declared because copied too many times
def log_g(z, data): # Computes log g(alpha, b) = log f(alpha, e^b) + b where f(alpha, beta) proportional to beta^{n-1} * prod((\beta^2 + (x_i - alpha)^2))
    alpha, b = z[0], z[1]
    beta = np.exp(b)
    n = len(data)
    
    # Log-likelihood + log-prior
    log_lik = -np.sum(np.log(beta**2 + (data - alpha)**2))
    log_prior = (n - 1) * b + b  # (n-1)*b from f(alpha, e^b) and +b
    return log_lik + log_prior

# run chains for all s...
step_sizes = [0.02, 0.5, 5.0]
results = {}

plt.figure()
for i, s in enumerate(step_sizes):
    T=50000
    chain = np.zeros((T, 2))  # stores (alpha, b)
    current_z = np.array([0.0, 0.0]) # start at (0,0)
    current_log_g = log_g(current_z, scans)
    accepts = 0
    
    for t in range(T):
        # propose using standard 2D Gaussian scaled by s
        proposal = current_z + s * np.random.normal(size=2)
        proposal_log_g = log_g(proposal, scans)
        
        # acceptance step
        if np.log(np.random.uniform()) < (proposal_log_g - current_log_g):
            current_z = proposal
            current_log_g = proposal_log_g
            if t > 0: accepts += 1
            
        chain[t] = current_z
        
    chain, acceptance_rate = chain, accepts / (T - 1)
    
    results[s] = {'chain': chain, 'acc': acceptance_rate} # map that!
    print(f"Acceptance Rate of s = {s}: {acceptance_rate:.2%}\n")
    # Plot first 400 steps of alpha
    plt.subplot(3, 1, i+1)
    plt.plot(chain[:400, 0])
    plt.title(f"First 400 steps of alpha, Step Size s = {s}")
    plt.grid(True, alpha=0.3)
    plt.ylabel("alpha")
plt.tight_layout()
plt.show()

# optimal chain - s = .5
best_s = 0.5
best_chain = results[best_s]['chain'][5000:] # Discard first 5,000 steps...
mcmc_alpha = best_chain[:, 0]
mcmc_beta = np.exp(best_chain[:, 1])

# exact outcome from grid
alpha_grid = np.linspace(0.5, 2.5, 200)
b_grid = np.linspace(-1.5, 1.0, 200)
A, B_log = np.meshgrid(alpha_grid, b_grid)
Z_grid = np.zeros_like(A)

for r in range(A.shape[0]):
    for c in range(A.shape[1]):
        Z_grid[r, c] = np.exp(log_g([A[r, c], B_log[r, c]], scans))

# normalize the joint grid density.....
dx = (alpha_grid[1] - alpha_grid[0]) * (b_grid[1] - b_grid[0])
Z_norm = Z_grid / np.sum(Z_grid)

# comp marginal distribution expectations from Grid
grid_alpha_mean = np.sum(A * Z_norm)
grid_beta_mean = np.sum(np.exp(B_log) * Z_norm)

# grid confidence intervals....
p_levels=[0.05, 0.95]
# alpha
axis_to_sum = 0
marginal = np.sum(Z_norm, axis=axis_to_sum)
cdf = np.cumsum(marginal) / np.sum(marginal)
grid_alpha_ci = [alpha_grid[np.searchsorted(cdf, p)] for p in p_levels]

# beta
axis_to_sum = 1
marginal = np.sum(Z_norm, axis=axis_to_sum)
cdf = np.cumsum(marginal) / np.sum(marginal)
grid_beta_ci = [np.exp(b_grid)[np.searchsorted(cdf, p)] for p in p_levels]

# tabulate!
print(f"Sim alpha avg: {np.mean(mcmc_alpha):.4f} \n  90% Confidence Interval: [{np.percentile(mcmc_alpha, 5):.4f}, {np.percentile(mcmc_alpha, 95):.4f}]")
print(f"Grid alpha avg: {grid_alpha_mean:.4f}  \n  90% Confidence Interval: [{grid_alpha_ci[0]:.4f}, {grid_alpha_ci[1]:.4f}]")
print(f"Sim beta avg:  {np.mean(mcmc_beta):.4f}  \n  90% Confidence Interval: [{np.percentile(mcmc_beta, 5):.4f}, {np.percentile(mcmc_beta, 95):.4f}]")
print(f"Grid beta avg:  {grid_beta_mean:.4f} \n  90% Confidence Interval: [{grid_beta_ci[0]:.4f}, {grid_beta_ci[1]:.4f}]")
print(f"Sample avg:     {np.mean(scans):.4f}  \n  Sample med: {np.median(scans):.4f}\n")

# density contours vs samples...
plt.figure()
plt.contour(A, B_log, Z_norm, levels=15)
plt.scatter(best_chain[:, 0], best_chain[:, 1], s=1)
plt.title(f"Posterior Joint Distribution (s={best_s})")
plt.xlabel("alpha")
plt.ylabel("log beta")
plt.grid(True, alpha=0.2)
plt.show()


# Q1.d

true_alpha, true_beta = 1.5, 0.8
n_sizes = [10, 100, 1000]
num_datasets = 2000

print("Q1.D : ESTIMATOR SCATTER PERFORMANCE")
for n in n_sizes:
    # vectorized generation of 2000 datasets of size n from Cauchy(alpha, beta)
    samples = cauchy.rvs(loc=true_alpha, scale=true_beta, size=(num_datasets, n)) # get 2000 n size sets from cauchy(alpha,beta
    
    means = np.mean(samples, axis=1)
    medians = np.median(samples, axis=1)
    
    # get spread metrics from IQ ranges...
    iqr_mean = (np.percentile(means, 75) - np.percentile(means, 25)) / 1.349
    iqr_median = (np.percentile(medians, 75) - np.percentile(medians, 25)) / 1.349
    
    print(f"n = {n:4d} , Sample Avg Scatter: {iqr_mean:8.4f} , Sample Med Scatter: {iqr_median:8.4f}")


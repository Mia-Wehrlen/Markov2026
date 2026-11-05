import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# params...
np.random.seed(666) # reproducable baybe
days = 30 
hours_per_day = 24
T = days * hours_per_day  # total num hours to sim..
lambda_max = 20.0

def true_lambda(t): # baby helper to make this easier to type out
    return 12.0 + 8.0 * np.cos(np.pi * (t - 16.0) / 12.0)

# ----- GENERATE BY THINNING... -----

n_proposals = np.random.poisson(lambda_max * T) # get the number of proposals using lambda max...
proposal_times = np.sort(np.random.uniform(0, T, n_proposals)) # draw uniforms over 0 to T and sort by bigness to get ordered times Ti...

# now THIN
acceptance_probabilities = true_lambda(proposal_times) / lambda_max # calc the likelyhood of each point...
kept_mask = np.random.uniform(0, 1, n_proposals) < acceptance_probabilities # discard those that are too small...
kept_times = proposal_times[kept_mask] # use mask to keep some percentage of drawn times....

# how many (fractionally) where kept?? --- should be close to the prob of being accepted...
frac_kept = len(kept_times) / n_proposals
print("Theoretical fraction expected: 0.6000")
print(f"Actual fraction of proposed points kept: {frac_kept:.4f}\n")

# ----- OK NOW -----
# plot mean number of arrivals in each clock hour per day with true lambda(t)...
# Aggregate arrivals into structural clock hours per day
clock_hours = kept_times % 24 # this produces what 'clock hour' the time is in... basically it discretizes it...
hour_counts, bin_edges = np.histogram(clock_hours, bins=np.arange(25))
mean_hour_arrivals = hour_counts / days
t_plot = np.linspace(0, 24, 200) # get a linspace over the plot for lambda ploting

plt.figure(figsize=(10, 6))
plt.bar(np.arange(24), mean_hour_arrivals, label='Simulated Mean')
plt.plot(t_plot, true_lambda(t_plot), color='green', label='True $\lambda(t)$')
plt.xlabel('Hours After Midnight')
plt.ylabel('Mean Arrivals')
plt.legend()
plt.tight_layout()
plt.show()


# 3->4 am predictions...

counts_3_to_4 = []
for day in range(days): # for each day....
    t_start = day * hours_per_day + 3 # start is 3 am of day*24 hours bc each day starts over at midnight...
    t_end = t_start + 1 # there is a 1 hour gap btw these two times... 
    count = np.sum(1 for x in kept_times if(x >= t_start) & (x < t_end)) # total num of times btw these two times... 
    counts_3_to_4.append(count)

print("--- Tabulate 3:00 AM to 4:00 AM  ---")
print("Theoretical Poisson Param: 4.091")
print(f"Sim Mean: {np.mean(counts_3_to_4):.3f}, Theoretical Mean: 4.091")
print(f"Sim Variance: {np.var(counts_3_to_4, ddof=1):.3f}, Theoretical Var: 4.091\n")

# max log-liklihod compared with const rate n/T...

def neg_log_likelihood(params, event_times, T_total): # returns the inverse log likelihood... 
    a, b, phi = params
    
    if a <= 0 or b < 0 or a < b:
        return 1e10
        
    # calc instant rate values across all events
    lambdas = a + b * np.cos(np.pi * (event_times - phi) / 12.0)
    if np.any(lambdas <= 0):
        return 1e10

    Lambda_total = a * T_total
    
    return -(np.sum(np.log(lambdas)) - Lambda_total)

initial_guess = [10.0, 5.0, 14.0]
mle_results = opt.minimize(neg_log_likelihood, initial_guess, args=(kept_times, T), method='Nelder-Mead')
a_fit, b_fit, phi_fit = mle_results.x

# calc baseline homogeneous log-likelihood for comparison
n_events = len(kept_times)
constant_rate = n_events / T
constant_log_lik = n_events * np.log(constant_rate) - constant_rate * T

print("--- Maximum Likelihood Estimation ---")
print(f"Fitted Parameters -> a: {a_fit:.2f} (True: 12.0), b: {b_fit:.2f} (True: 8.0), phi: {phi_fit:.2f} (True: 16.0)")
print(f"Maximized Log-Likelihood: {-mle_results.fun:.2f}")
print(f"Constant Rate Log-Likelihood: {constant_log_lik:.2f}\n")

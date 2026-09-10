#
# HW2_Problems_3_4_5.py
#
# inverse sampling
# 
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(666)  # reproducable baybe - >:)

# Question 3!!!
rng = np.random.default_rng()

# Acceptance-Rejection sampling
# folded normal f(x) ~ sqrt(2/pi)*exp(-x^2/2), x>=0
N = 10**4 # number of samples
lam = .5 # exponential rate
c = 4/np.e # envelope constant

samples = np.zeros(N) # storage
i = 0
count = 0
last_time = time.perf_counter()
temp_time =  0
avg_time = 0
while i < N:
    # Proposal: exponential
    Y = -np.log(rng.random())/lam
    # Target density f(Y)
    fY = Y*np.exp(-Y)
    # Proposal density g(Y)
    gY = lam*np.exp(-lam*Y)
    # Acceptance test
    if rng.random() <= fY/(c*gY):
        samples[i] = Y
        i += 1
        temp_time = time.perf_counter()
        avg_time += temp_time - last_time
    # track samples till 
    count +=1

avg_time = avg_time/N
print(f"Empirical Acceptance Fraction: {N/count} (1/c: {1/c})\n")
print(f"Mean Time per Accepted Sample: {avg_time}\n")

# Compare empirical histogram with true folded Gaussian
x = np.linspace(0, 5, 400)
f = x*np.exp(-x)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.title('lambda = .5')
plt.hist(samples, bins='fd', density=True, label='Empirical pdf')
plt.plot(x, f, 'r-', linewidth=2, label='True distribution')
plt.xlim([0, 5])
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(labelsize=18)
plt.xlabel('$x$', fontsize=24)
plt.ylabel('$f(x)$', fontsize=24)
plt.tight_layout()
plt.show()


# again with different lambda...
lam = .2 # exponential rate
c = 2.2993 # envelope constant

samples = np.zeros(N) # storage
i = 0
count = 0
last_time = time.perf_counter()
temp_time =  0
avg_time = 0
while i < N:
    # Proposal: exponential
    Y = -np.log(rng.random())/lam
    # Target density f(Y)
    fY = Y*np.exp(-Y)
    # Proposal density g(Y)
    gY = lam*np.exp(-lam*Y)
    # Acceptance test
    if rng.random() <= fY/(c*gY):
        samples[i] = Y
        i += 1
        temp_time = time.perf_counter()
        avg_time += temp_time - last_time
    # track samples till 
    count +=1

avg_time = avg_time/N
print(f"Empirical Acceptance Fraction: {N/count} (1/c: {1/c})\n")
print(f"Mean Time per Accepted Sample: {avg_time}\n")
        
# Compare empirical histogram with true folded Gaussian
x = np.linspace(0, 5, 400)
f = x*np.exp(-x)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.title('lambda=.2')
plt.hist(samples, bins='fd', density=True, label='Empirical pdf')
plt.plot(x, f, 'r-', linewidth=2, label='True distribution')
plt.xlim([0, 5])
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(labelsize=18)
plt.xlabel('$x$', fontsize=24)
plt.ylabel('$f(x)$', fontsize=24)
plt.tight_layout()
plt.show()

# Question 4!!!

# params...
alpha = 0.9
lambda_f = 1000
lambda_s = 10
N = 10**5

# sampling
U = rng.uniform(size=N)
V = rng.uniform(size=N)
mask = U < alpha
T = np.where(mask, -np.log(V) / lambda_f, -np.log(V) / lambda_s) # mask that shit....

# simulated stats...
sim_mean = T.mean()
threshold = 50/1000  # 50 ms in seconds
sim_prob = np.mean(T > threshold)

# calculated values....
Ex = alpha / lambda_f + (1 - alpha) / lambda_s
calc_prob = alpha * np.exp(-lambda_f * threshold) + (1 - alpha) * np.exp(-lambda_s * threshold)

print(f"Empirical Mean: {sim_mean*1000} ms (Theoretical Mean: {Ex*1000} ms)")
print(f"Empirical P(T>50ms): {sim_prob} (Theoretical P(T>50ms): {calc_prob})")

# calculated pdf...
t = np.logspace(np.log10(1e-5), np.log10(1.0), 2000)
f = alpha * lambda_f * np.exp(-lambda_f * t) + (1 - alpha) * lambda_s * np.exp(-lambda_s * t)

# plot the stuff...
bins = np.logspace(np.log10(1e-5), np.log10(1.0), 120) # force bins to be log....

plt.figure(1, figsize=(8, 6))
plt.hist(T,bins=bins, density=True, alpha=0.6, color='blue', label='Empirical pdf')
plt.plot(t, f, color='red', lw=2, label='True pdf')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=12)
plt.xlim([1e-5, 1.0])
plt.ylim([1e-2, 1e4])
plt.tight_layout()
plt.savefig('dwell_time_histogram.png', dpi=150)
plt.show()

# Question 5!!!
# sample h by inversion......
U = np.random.uniform(0, 1, 10**5)
H_inv = 1 - np.sqrt(U) # H^-1 = 1- sqrt of U
print(f"Empirical Mean(inverse samp): {np.mean(H_inv):.4f} (Theoretical Mean: 0.3333)\n")

# simulation section...
N = 10**4
R = 10**3   # num realizations

# init as oness....
C = np.ones(R, dtype=float)

# step over N elements for R draws each iter...
for n in range(3, N):
    prob = C / n
    draw = np.random.uniform(0, 1, R) < prob
    C += draw

# calculate sim core ratios
z_ratio = C / N

# Metrics for comparison
empirical_mean = np.mean(z_ratio)
empirical_sd = np.std(z_ratio)
calc_ratio = empirical_sd / empirical_mean
smallest = int(np.min(C))
largest = int(np.max(C))

print(f"Empirical Mean(simulated): {empirical_mean:.4f} (Theoretical Mean: 0.3333)")
print(f"Empirical SD/Mean Ratio: {calc_ratio:.4f} (Theoretical SD/Mean Ratio: 0.7071)")
print(f"Smallest Core Size : {smallest}")
print(f"Largest Core Size : {largest}")

# theoretical density function h(z) = 2(1 - z)
z = np.linspace(0, 1, 500)
h = 2 * (1 - z)

# plot normalized histogram of simulated data
plt.figure(2, figsize=(8, 6))
plt.clf()
plt.hist(z_ratio, bins=40, density=True, alpha=0.6, color='blue', label='Empirical pdf')
plt.plot(z, h, color='red', linewidth=2, label='True pdf')
plt.xlim([0, 1])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=16)
plt.tick_params(labelsize=18)
plt.xlabel('$z = C/N$', fontsize=24)
plt.tight_layout()
plt.savefig('core_size.png', dpi=150)
plt.show()

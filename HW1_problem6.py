#
# HW1_Problem6.py
#
# Run some large number of samples to show convergence to real value as n >>
#
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)  # reproducable baybe - >:)
rng = np.random.default_rng()

# constants! 
real = 23 * np.pi / 192 # calculated result...
# log time sample space...
Nsim = np.logspace(2, 6, num=50, dtype=int) # 50 total 'datasets'

a = 0
b = 1

# Monte Carlo -- sample from uniforms....
results = []
for n in Nsim:
    # # Option 1: use the built in uniform fxn in numpy...
    # X = np.random.uniform(0, 1, n) # gen N uniform samples
    # Y = np.random.uniform(0, 1, n) # gen N uniform samples
    # Z = np.random.uniform(0, 1, n) # gen N uniform samples

    # OORRR -- I think this is what the assignment wants but it makes me sad to look at....
    X = a + (b - a)*rng.random(n)
    Y = a + (b - a)*rng.random(n)
    Z = a + (b - a)*rng.random(n)
    
    # conditionals....
    P1 = (X**2 + Y**2) < Z
    P2 = (Z**2) > (X * Y)
    
    # estimate probability
    victory = np.mean(P1 & P2)
    results.append(victory)

# Plot the Results
plt.figure()
plt.plot(Nsim, results, label='Simulated Estimate')
plt.axhline(y=real, color='r', label=f'Calculated Actual ({real:.6f})')

plt.xscale('log')
plt.xlabel('Num Samples')
plt.ylabel('Simulated P')
plt.title('Monte Carlo Simulation')
plt.grid(True, which="both", alpha=0.6)
plt.legend()
plt.show()

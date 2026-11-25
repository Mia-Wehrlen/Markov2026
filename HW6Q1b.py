import numpy as np

def analyze_wielandt_chain(m):
    # gen the m x m transition matrix P
    P = np.zeros((m, m))
    for i in range(m - 1):
        P[i, i + 1] = 1.0
    P[m - 1, 0] = 0.5
    P[m - 1, 1] = 0.5
    
    # track the zero pattern to find the smallest power M where P^M > 0
    # B behaves as a boolean/adjacency mask tracking reachability
    B = (P > 0).astype(int)
    M = 1
    while not np.all(B > 0):
        B = ((B @ (P > 0)) > 0).astype(int)
        M += 1
            
    Compute eigenvalues to find the second largest modulus
    eigenvalues = np.linalg.eigvals(P)
    
    # sort absolute magnitudes in small to big....
    sorted_moduli = sorted([np.abs(lam) for lam in eigenvalues], reverse=True)
    
    # grab the second largest!
    lambda_2_modulus = sorted_moduli[1]
    
    return M, lambda_2_modulus # the spoils

# print comparison table for m = 3 through 12
print(f"{'m':>2} | {'M':>3} | {'wielandt Bound':>14} | {'|λ₂|':>8}")
print("-" * 37)
for m in range(3, 13):
    M, lam2 = analyze_wielandt_chain(m)
    wielandt_bound = (m - 1)**2 + 1
    print(f"{m:2d} | {M:3d} | {wielandt_bound:14d} | {lam2:.4f}")

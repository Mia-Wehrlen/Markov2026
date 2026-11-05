import numpy as np

# params
np.random.seed(666) # reproduceable baybe
N = 2*10**5
T = 5  # time window

# gen at given rate and then thin...
star_emitted = np.random.poisson(3 * T, N)
sky_emitted = np.random.poisson(2 * T, N)
star_det_thinned = np.random.binomial(star_emitted, 0.4)
sky_det_thinned = np.random.binomial(sky_emitted, 0.4)
total_det_thinned = star_det_thinned + sky_det_thinned

# generate at relative rates :)
star_det_gen = np.random.poisson(1.2 * T, N)
sky_det_gen = np.random.poisson(0.8 * T, N)
total_det_gen = star_det_gen + sky_det_gen

# tabulate counts!
print("--- Tabulated Counts, Means and Vars ---")
print(f"(Thinned) Star Count - Mean: {np.mean(star_det_thinned):.4f}, Variance: {np.var(star_det_thinned):.4f}")
print(f"(Gen) Star Count  - Mean: {np.mean(star_det_gen):.4f}, Variance: {np.var(star_det_gen):.4f}")
print(f"(Thinned) Sky Count  - Mean: {np.mean(sky_det_thinned):.4f}, Variance: {np.var(sky_det_thinned):.4f}")
print(f"(Gen) Sky Count   - Mean: {np.mean(sky_det_gen):.4f}, Variance: {np.var(sky_det_gen):.4f}")

corr_thinned = np.corrcoef(star_det_thinned, sky_det_thinned)[0, 1]
corr_gen = np.corrcoef(star_det_gen, sky_det_gen)[0, 1]

corr_star = np.corrcoef(star_det_thinned, star_det_gen)[0, 1]
corr_sky = np.corrcoef(sky_det_thinned, sky_det_gen)[0, 1]

print("\nTBH: the language here is kind vauge so....")
print(f"Correlation between Star and Sky Counts (thinned): {corr_thinned:.4f} (Theoretical: 0.0000)")
print(f"Correlation between Star and Sky Counts (gen): {corr_gen:.4f} (Theoretical: 0.0000)")
print(f"Correlation between Thinned Star and Gen Star Counts: {corr_star:.4f}")
print(f"Correlation between Thinned Sky and Gen Sky Counts: {corr_sky:.4f}")


print(f"\nTotal Detections Mean (thinned): {np.mean(total_det_thinned):.4f}, Variance: {np.var(total_det_thinned):.4f} (Theoretical Poisson(10) Var: 10.0)")
print(f"Total Detections Mean (gen): {np.mean(total_det_gen):.4f}, Variance: {np.var(total_det_gen):.4f} (Theoretical Poisson(10) Var: 10.0)")


# Conditional estimation
cond_6_star = star_det_thinned[total_det_thinned == 10]
p_6_star_est = np.mean(cond_6_star == 6)
print(f"\nEstimated P(exactly 6 Star Detections | Total = 10) (thinned): {p_6_star_est:.4f} (Theoretical: 0.2508)")

cond_6_star = star_det_gen[total_det_gen == 10]
p_6_star_est = np.mean(cond_6_star == 6)
print(f"\nEstimated P(exactly 6 Star Detections | Total = 10) (gen): {p_6_star_est:.4f} (Theoretical: 0.2508)")

# --- 3. First Detection Times Simulation ---
X = np.random.exponential(1 / 1.2, N)
Y = np.random.exponential(1 / 0.8, N)

star_first = X < Y
first_time = np.minimum(X, Y)
both_detected_time = np.maximum(X, Y)
type_first = star_first.astype(int)  # 1 if star first, 0 if sky first

p_star_first_est = np.mean(star_first)
p_star_first_within_1_est = np.mean(star_first & (first_time <= 1))
e_M_est = np.mean(both_detected_time)
corr_time_type = np.corrcoef(first_time, type_first)[0, 1]

print("\n--- Tabulated First Arival Time Probs ---")
print("\n NOTE: I am assuming here that this means: \n'draw some given arival time for each type, and estimate the probablities from there' \nand NOT: \n'use the arrival times of the previous draws to estimate this for each method'")
print(f"\nEstimated P(Star First): {p_star_first_est:.4f} (Theoretical: 0.6000)")
print(f"Estimated P(Star First and within 1s): {p_star_first_within_1_est:.4f} (Theoretical: 0.5188)")
print(f"Estimated E[M] (Time to detect both): {e_M_est:.4f} s (Theoretical: 1.5833 s)")
print(f"Correlation between first arrival time and its type: {corr_time_type:.4f} (Theoretical: 0.0000)")

import numpy as np
import pandas as pd
import seaborn as sns

import numpy as np
import pandas as pd

# load data and setup teams
df = pd.read_csv('big12.csv')
teams = sorted(list(set(df['home']).union(set(df['away'])))) # get the list of team names...
m = len(teams) 
team_to_idx = {team: i for i, team in enumerate(teams)} #use names as indexes...

def h(x): # helper to calc h(x) where h() = (1/2) +(1/2)sgn(x-(1/2)ssqrt(abs(2x-1))
    return 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(np.abs(2*x - 1))

# get 2025 and 2026 game history
df_2025 = df[df['season'] == 2025] # assume all games completed form last year...
df_2026 = df[(df['season'] == 2026) & (df['completed'] == 1)]# only want completed games for current year

# compute 2026 observed games counts
n = np.zeros(m) # vector of num games of size teams...
for _, row in df_2026.iterrows():
    n[team_to_idx[row['home']]] += 1 # a home game
    n[team_to_idx[row['away']]] += 1 # an away game....

# gen observed outcomes for 2026 so far.... 
H_2026 = np.zeros((m, m))
for _, row in df_2026.iterrows(): # iterate over the rows..
    i, j = team_to_idx[row['home']], team_to_idx[row['away']]
    S, T = row['home_pts'], row['away_pts']
    H_2026[i, j] += h((S + 1) / (S + T + 2))
    H_2026[j, i] += h((T + 1) / (S + T + 2))

# def the Prior Matrices (P_flat, P_last, P_poll)
P_flat = np.ones((m, m)) * 0.5
np.fill_diagonal(P_flat, 0)

# construct based on last season....
P_last = np.ones((m, m)) * 0.5
meetings_2025 = np.zeros((m, m))
sums_2025 = np.zeros((m, m))
for _, row in df_2025.iterrows():
    i, j = team_to_idx[row['home']], team_to_idx[row['away']]
    S, T = row['home_pts'], row['away_pts']
    sums_2025[i, j] += h((S + 1) / (S + T + 2))
    sums_2025[j, i] += h((T + 1) / (S + T + 2))
    meetings_2025[i, j] += 1
    meetings_2025[j, i] += 1

for i in range(m):
    for j in range(m):
        if meetings_2025[i, j] > 0:
            P_last[i, j] = sums_2025[i, j] / meetings_2025[i, j]
np.fill_diagonal(P_last, 0)

# preseason Poll Prior --- draw from index i ... 
poll_scores = {t: 100 - 4*idx for idx, t in enumerate(teams)}  
P_poll = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        if i != j:
            si, sj = poll_scores[teams[i]], poll_scores[teams[j]]
            P_poll[i, j] = si / (si + sj)

# solve for rankings... :(
w = 1
priors = {'Flat': P_flat, 'Last Season': P_last, 'Preseason Poll': P_poll} # map that shit!
results_dict = {'Team': teams} # index by teams for later plotting...

for name, P in priors.items(): # for each prior...
    A = np.zeros((m, m)) #init A 0
    for i in range(m):
        for j in range(m):
            if i != j:
                A[i, j] = (H_2026[i, j] + w * P[i, j]) / (n[i] + w * (m - 1)) # calc A with aij vals... aij = (sum over i v j games of hij + w P(ii,j))/(ni +w(m-1))
                
    eigenvalues, eigenvectors = np.linalg.eig(A) # get eigens....
    idx_sort = np.argsort(np.abs(eigenvalues))[::-1] # sort by abs size...
    
    lambda_star = eigenvalues[idx_sort[0]].real # get biggest....
    lambda_2 = eigenvalues[idx_sort[1]] # second  biggest
    
    r = eigenvectors[:, idx_sort[0]].real
    r = r / np.sum(r)  # Normalize
    
    results_dict[name] = r
    print(f"{name}")
    print(f"λ*  = {lambda_star:.4f}")
    print(f"|λ2/λ*| = {np.abs(lambda_2)/lambda_star:.4f}\n")

# save to dataframe...
df_r = pd.DataFrame(results_dict)

# melt the df for ploting!!
df_melted = df_r.melt(id_vars='Team', var_name='Prior', value_name='Rank (r)')

# Initialize the plot layout
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid") #use seaborn bc it makes my life easy

# bar chart...
ax = sns.barplot( # seaborn parses for me...
    data=df_melted,
    x='Team',
    y='Rank (r)',
    hue='Prior', #cat by prior
)
plt.xticks(rotation=90)
plt.xlabel("Norm Team Strength (r)")
plt.ylabel("Teams")
plt.legend(title="Prior")
plt.tight_layout()

plt.show()

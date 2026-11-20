import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

df1 = pd.read_csv('results_Tree.csv')
tree_amts = pd.read_csv('tree_amounts.csv')

stress = df1[['iteration', 'obj_id', 't', 'species', 'temp_stress', 'light_stress', 'water_stress', 'is_dead']]
mortality = df1[['iteration', 'obj_id', 't', 'species', 'mort_prob', 'is_dead']]

# Stress by Species
oak_stress = stress.loc[stress['species'] == 'oak']
gum_stress = stress.loc[stress['species'] == 'gum']
wattle_stress = stress.loc[stress['species'] == 'wattle']

oak_stress_alive = oak_stress.loc[oak_stress['is_dead'] == False]
gum_stress_alive = gum_stress.loc[gum_stress['is_dead'] == False]
wattle_stress_alive = wattle_stress.loc[wattle_stress['is_dead'] == False]

# Avg Stress per Time Step per Species
avg_oak_stress = oak_stress_alive[['t', 'temp_stress', 'light_stress', 'water_stress']].groupby(['t'], as_index = False).mean()
avg_gum_stress = gum_stress_alive[['t', 'temp_stress', 'light_stress', 'water_stress']].groupby(['t'], as_index = False).mean()
avg_wattle_stress = wattle_stress_alive[['t', 'temp_stress', 'light_stress', 'water_stress']].groupby(['t'], as_index = False).mean()

# Seeing progess
print(tree_amts.head(5))
print(stress.head(5))
print(mortality.head(5))
print()

# Avg Stress 
print(avg_oak_stress.head(5))
print(avg_gum_stress.head(5))
print(avg_wattle_stress.head(5))

# Plots
# Oak (Random)
avg_oak_stress_melted = pd.melt(avg_oak_stress, id_vars='t', value_vars= ['temp_stress', 'light_stress', 'water_stress'], var_name='stress_type', value_name='stress_units')
# Line Plot: Avg Oak Stress over Time
sns.lineplot(data=avg_oak_stress_melted, x='t', y='stress_units', hue='stress_type')
plt.xlabel('Time Step (days)')
plt.ylabel('Stress Units')
plt.title('Average Quercus Stress')
plt.show()

# Gum -> Eucalyptus (Random)
avg_gum_stress_melted = pd.melt(avg_gum_stress, id_vars='t', value_vars= ['temp_stress', 'light_stress', 'water_stress'], var_name='stress_type', value_name='stress_units')
# Line Plot: Avg Gum Stress over Time
sns.lineplot(data=avg_gum_stress_melted, x='t', y='stress_units', hue='stress_type')
plt.xlabel('Time Step (days)')
plt.ylabel('Stress Units')
plt.title('Average Eucalyptus Stress')
plt.show()

# Wattle -> Acacia (Random)
avg_wattle_stress_melted = pd.melt(avg_wattle_stress, id_vars='t', value_vars= ['temp_stress', 'light_stress', 'water_stress'], var_name='stress_type', value_name='stress_units')
# Line Plot: Avg Oak Stress over Time
sns.lineplot(data=avg_wattle_stress_melted, x='t', y='stress_units', hue='stress_type')
plt.xlabel('Time Step (days)')
plt.ylabel('Stress Units')
plt.title('Average Acacia Stress')
plt.show()

# Mortality (Random)
mortality_alive = mortality.loc[mortality['is_dead'] == False]
avg_mort = mortality_alive[['t', 'mort_prob']].groupby(['t'], as_index=False).mean()
sns.lineplot(data=avg_mort, x='t', y='mort_prob')
plt.xlabel('Time Step (days)')
plt.ylabel('Probability')
plt.title('Average Mortality Probability Over Time')
plt.show()

# RDA
mort = mortality[['iteration', 't', 'is_dead']].groupby(['iteration', 't'], as_index=False).sum()
# print(mortality.head())
# Includes Iteration
rda = pd.merge(mort, tree_amts, on=['iteration', 't'], how='inner')
# Death Ratio Column Addition
rda['Death Ratio'] = rda['is_dead'] / rda['tree_amt']
print(rda.tail())

# Combined
total_tree = rda[['t', 'is_dead', 'tree_amt']].groupby(['t'], as_index = False).sum()
avg_rda = rda[['t', 'Death Ratio']].groupby(['t'], as_index = False).mean()

comb_rda = pd.merge(total_tree, avg_rda, on='t', how = 'inner')

print(comb_rda.tail())

# Mixed Graph of Death Ratio
sns.lineplot(data=comb_rda, x='t', y='Death Ratio')
plt.xlabel('Time Step (days)')
plt.ylabel('Death Ratio')
plt.title('Death Ratio Over Time')
plt.show()
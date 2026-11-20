import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Eucalyptus Control Group

gum = pd.read_csv('results_gum.csv')
gum_amts = pd.read_csv('gum_tree_amounts.csv')

# Stress/Mortality Split
gum_stress = gum[['iteration', 'obj_id', 't', 'temp_stress', 'light_stress', 'water_stress', 'is_dead']]
gum_mortality = gum[['iteration', 'obj_id', 't', 'mort_prob', 'is_dead']]

# Gum Stress of Alive
gum_stress_alive = gum_stress.loc[gum_stress['is_dead'] == False]
# Averaging
avg_gum_stress = gum_stress_alive[['t', 'temp_stress', 'light_stress', 'water_stress']].groupby(['t'], as_index = False).mean()

# Gum (Control)
avg_gum_stress_melted = pd.melt(avg_gum_stress, id_vars='t', value_vars= ['temp_stress', 'light_stress', 'water_stress'], var_name='stress_type', value_name='stress_units')
# Line Plot: Avg Gum Stress over Time
sns.lineplot(data=avg_gum_stress_melted, x='t', y='stress_units', hue='stress_type')
plt.xlabel('Time Step (days)')
plt.ylabel('Stress Units')
plt.title('Average Eucalyptus Stress (Control)')
plt.show()

# Gum RDA
mortality = gum_mortality[['iteration', 't', 'is_dead']].groupby(['iteration', 't'], as_index=False).sum()
# print(mortality.head())
# Includes Iteration
rda = pd.merge(mortality, gum_amts, on=['iteration', 't'], how='inner')
# Death Ratio Column Addition
rda['Death Ratio'] = rda['is_dead'] / rda['tree_amt']
print(rda.tail())

# Total Avg Trees throughout the Simulation
rda['Alive Trees'] = rda['tree_amt'] - rda['is_dead']

# Combined
total_tree = rda[['t', 'is_dead', 'tree_amt']].groupby(['t'], as_index = False).sum()
avg_rda = rda[['t', 'Death Ratio', 'Alive Trees']].groupby(['t'], as_index = False).mean()

comb_rda = pd.merge(total_tree, avg_rda, on='t', how = 'inner')

print(comb_rda.tail())

# Gum Graph of Death Ratio
sns.lineplot(data=comb_rda, x='t', y='Death Ratio')
plt.xlabel('Time Step (days)')
plt.ylabel('Death Ratio')
plt.title('Eucalyptus Death Ratio Over Time (Control)')
plt.show()

# Avg Alive Euca Over Time
sns.lineplot(data=comb_rda, x='t', y='Alive Trees')
plt.xlabel('Time Step (days)')
plt.ylabel('# of Trees')
plt.title('Average Amount of Eucalyptus Trees Over Time (Control)')
plt.show()

# Mortality (Control)
mortality_alive = gum_mortality.loc[gum_mortality['is_dead'] == False]
avg_mort = mortality_alive[['t', 'mort_prob']].groupby(['t'], as_index=False).mean()
sns.lineplot(data=avg_mort, x='t', y='mort_prob')
plt.xlabel('Time Step (days)')
plt.ylabel('Probability')
plt.title('Average Eucalyptus Mortality Probability Over Time (Control)')
plt.show()
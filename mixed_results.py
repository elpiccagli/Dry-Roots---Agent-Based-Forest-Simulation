import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

df1 = pd.read_csv('results_Tree.csv')
tree_amts = pd.read_csv('tree_amounts.csv')

stress = df1[['iteration', 'obj_id', 't', 'species', 'temp_stress', 'light_stress', 'water_stress', 'is_dead']]
mortality = df1[['iteration', 'obj_id', 't', 'species', 'mort_prob', 'is_dead']]
mortality['is_alive'] = ~mortality['is_dead']

# print(mortality.head())

oak_mort = mortality[['iteration', 'obj_id', 't', 'species', 'is_alive']].loc[mortality['species'] == 'oak']
euca_mort = mortality[['iteration', 'obj_id', 't', 'species', 'is_alive']].loc[mortality['species'] == 'gum']
wattle_mort = mortality[['iteration', 'obj_id', 't', 'species', 'is_alive']].loc[mortality['species'] == 'wattle']
alive = mortality[['iteration', 't', 'is_alive']].rename(columns={'is_alive': 'All Trees'}).groupby(['iteration', 't'], as_index=False).sum()

# print(oak_mort.head())
# print(alive.head())

oak_alive = oak_mort[['iteration', 't', 'is_alive']].rename(columns={'is_alive': 'Quercus'}).groupby(['iteration', 't'], as_index=False).sum()
euca_alive = euca_mort[['iteration', 't', 'is_alive']].rename(columns={'is_alive': 'Eucalyptus'}).groupby(['iteration', 't'], as_index=False).sum()
wattle_alive = wattle_mort[['iteration', 't', 'is_alive']].rename(columns={'is_alive': 'Acacia'}).groupby(['iteration', 't'], as_index=False).sum()

# print(oak_alive.head())

# print(len(alive), len(oak_alive), len(euca_alive), len(wattle_alive), sep=', ')

total_oak = pd.merge(alive, oak_alive, how='inner', on=['iteration', 't'])
euca_wattle = pd.merge(euca_alive, wattle_alive, how='inner', on=['iteration', 't'])
total_alive = pd.merge(total_oak, euca_wattle, how='inner', on=['iteration', 't'])

print(total_alive.tail())
print(len(total_alive))

avg_alive = total_alive[['t', 'Quercus', 'Eucalyptus', 'Acacia']].groupby(['t'], as_index=False).mean()

avg_total_trees = pd.melt(avg_alive, id_vars='t', value_vars= ['Quercus', 'Eucalyptus', 'Acacia'], var_name='Tree Type', value_name='tree_amt')

# Mixed Avg Trees Over Time
sns.lineplot(data=avg_total_trees, x='t', y='tree_amt', hue='Tree Type')
plt.xlabel('Time Step (days)')
plt.ylabel('# of Trees')
plt.title('Average Amount of Trees Over Time')
plt.show()
import pandas as pd
from scipy.stats import f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# Mixed
mixed = pd.read_csv('results_Tree.csv')
mixed_amts = pd.read_csv('tree_amounts.csv')
mixed_end = mixed.loc[mixed['t'] == 1825][['iteration', 'is_dead']].groupby(['iteration']).sum()
mixed_end['experiment'] = 'mixed'
mixed_end_amts = mixed_amts.loc[mixed_amts['t']==1825]
mixed_df = pd.merge(mixed_end, mixed_end_amts, on='iteration', how='inner')

# Oak
oak = pd.read_csv('results_oak.csv')
oak_amts = pd.read_csv('oak_tree_amounts.csv')
oak_end = oak.loc[oak['t'] == 1825][['iteration', 'is_dead']].groupby(['iteration']).sum()
oak_end['experiment'] = 'oak'
oak_end_amts = oak_amts.loc[oak_amts['t']==1825]
oak_df = pd.merge(oak_end, oak_end_amts, on='iteration', how='inner')

# Gum
gum = pd.read_csv('results_gum.csv')
gum_amts = pd.read_csv('gum_tree_amounts.csv')
gum_end = gum.loc[gum['t'] == 1825][['iteration', 'is_dead']].groupby(['iteration']).sum()
gum_end['experiment'] = 'gum'
gum_end_amts = gum_amts.loc[gum_amts['t']==1825]
gum_df = pd.merge(gum_end, gum_end_amts, on='iteration', how='inner')

# Wattle
wattle = pd.read_csv('results_wattle.csv')
wattle_amts = pd.read_csv('wattle_tree_amounts.csv')
wattle_end = wattle.loc[wattle['t'] == 1825][['iteration', 'is_dead']].groupby(['iteration']).sum()
wattle_end['experiment'] = 'wattle'
wattle_end_amts = wattle_amts.loc[wattle_amts['t']==1825]
wattle_df = pd.merge(wattle_end, wattle_end_amts, on='iteration', how='inner')

# Need to Get Final Amounts for it all
mortality = pd.concat([mixed_df, oak_df, gum_df, wattle_df], ignore_index=True)
mortality_summary = mortality[['experiment','iteration', 'is_dead', 'tree_amt']]
mortality_summary['mortality_ratio'] = mortality_summary['is_dead'] / mortality_summary['tree_amt']

# print(mortality_summary.head())
# print()
# print(mortality_summary.tail())

mixed = mortality_summary.loc[mortality_summary['experiment'] == 'mixed']['mortality_ratio']
oak = mortality_summary.loc[mortality_summary['experiment'] == 'oak']['mortality_ratio']
gum = mortality_summary.loc[mortality_summary['experiment'] == 'gum']['mortality_ratio']
wattle = mortality_summary.loc[mortality_summary['experiment'] == 'wattle']['mortality_ratio']

anova_result = f_oneway(mixed, oak, gum, wattle)

print(anova_result)
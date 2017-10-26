# %load q01_plot_corr/build.py
# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:
def plot_corr(data,size=11):
    plt.figure(figsize=(size,size))
    sns.heatmap(data.corr(),cmap='YlOrRd')

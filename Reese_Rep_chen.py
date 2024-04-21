import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Read data
cz_all = pd.read_csv(r"C:\Users\Owner\Desktop\research\spring 24\asset-pricing-replications\data\PredictorPortsFull.csv")

# Read signal document data
signaldoc = pd.read_csv(r'C:\Users\Owner\Desktop\research\spring 24\asset-pricing-replications\data\SignalDoc.csv')
signaldoc = pd.read_csv(r'C:\Users\Owner\Desktop\research\spring 24\asset-pricing-replications\data\SignalDoc.csv') \
    .rename(columns={'Acronym': 'signalname'}) \
    .assign(pubdate=lambda x: pd.to_datetime(x['Year'].astype(str) + '-12-31'),
            sampend=lambda x: pd.to_datetime(x['SampleEndYear'].astype(str) + '-12-31'),
            sampstart=lambda x: pd.to_datetime(x['SampleStartYear'].astype(str) + '-01-01')) \
    .drop(columns=['Notes', 'Detailed Definition'])

# Filter and manipulate data
czret = cz_all.loc[(cz_all['ret'].notna()) & (cz_all['port'] == 'LS')]
czret = czret.merge(signaldoc, how='left')

# Define sample types
czret['samptype'] = np.select(
    [
        (czret['date'] >= czret['sampstart']) & (czret['date'] <= czret['sampend']),
        (czret['date'] > czret['sampend']) & (czret['date'] <= czret['sampend'] + pd.DateOffset(months=36)),
        (czret['date'] > czret['pubdate'])
    ],
    ['in-samp', 'out-of-samp', 'post-pub'],
    default='NA_character_'
)

czret = czret[['signalname', 'date', 'ret', 'samptype', 'sampstart', 'sampend']]

# Pivot and create correlation matrix
czretmat = czret.pivot_table(index='date', columns='signalname', values='ret')
cormat = czretmat.corr(method='pearson').values
corlong = cormat[np.tril_indices(cormat.shape[0], k=-1)]

# Plot histogram of correlation
plt.hist(corlong, bins=50, alpha=0.8, color='blue')
plt.xlabel('Pairwise Corr Between Monthly Returns')

plt.ylabel('Count')
plt.savefig(r"C:\Users\Owner\Desktop\research\spring 24\asset-pricing-replications\reeserepresults\correlation.pdf", format='pdf', bbox_inches='tight')
plt.show()
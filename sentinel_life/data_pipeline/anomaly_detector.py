import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/tn_ndvi_multiyear.csv')
print(f'Loaded: {len(df)} records')

baseline = df.groupby(['district','month'])['ndvi'].mean().reset_index()
baseline.columns = ['district','month','ndvi_baseline']

df = df.merge(baseline, on=['district','month'])
df['anomaly_pct'] = ((df['ndvi'] - df['ndvi_baseline']) / df['ndvi_baseline'] * 100).round(2)

stress = df[df['anomaly_pct'] < -10].sort_values('anomaly_pct')
print(f'Stress events found: {len(stress)}')
print(stress[['district','year','month','anomaly_pct']].to_string())

df.to_csv('data/processed/tn_ndvi_anomaly.csv', index=False)
print('Saved!')
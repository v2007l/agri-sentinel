import pandas as pd
import numpy as np

# Load anomaly data
df = pd.read_csv('data/processed/tn_ndvi_anomaly.csv')
print(f'Before cleaning: {len(df)} records')

# Remove -100% (cloud cover / no satellite data)
df_clean = df[df['ndvi'] > 0.05].copy()
print(f'After cleaning: {len(df_clean)} records')

# Real stress events only
stress = df_clean[df_clean['anomaly_pct'] < -10].sort_values('anomaly_pct')
print(f'\n⚠️  REAL Stress events: {len(stress)}')
print(stress[['district','year','month','ndvi','anomaly_pct']].to_string())

# Top 5 most stressed districts
print('\n🚨 TOP 5 MOST STRESSED DISTRICTS:')
top5 = df_clean.groupby('district')['anomaly_pct'].mean().sort_values().head(5)
print(top5.to_string())

# Save clean data
df_clean.to_csv('data/processed/tn_ndvi_clean.csv', index=False)
print('\n✅ Clean data saved!')
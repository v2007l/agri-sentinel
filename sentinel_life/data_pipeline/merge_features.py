import pandas as pd
import numpy as np

# Load both datasets
ndvi = pd.read_csv('data/processed/tn_ndvi_clean.csv')
rain = pd.read_csv('data/processed/tn_rainfall.csv')

print(f'NDVI records: {len(ndvi)}')
print(f'Rainfall records: {len(rain)}')

# Merge on district + year + month
df = ndvi.merge(rain, on=['district','year','month'], how='inner')
print(f'Merged records: {len(df)}')

# Rainfall anomaly
rain_baseline = df.groupby(['district','month'])['rainfall_mm'].transform('mean')
df['rainfall_anomaly_pct'] = ((df['rainfall_mm'] - rain_baseline) / rain_baseline * 100).round(2)

# Compound stress score
df['compound_stress'] = (
    df['anomaly_pct'].clip(-100, 0).abs() * 0.6 +
    df['rainfall_anomaly_pct'].clip(-100, 0).abs() * 0.4
).round(2)

# Top stressed districts
print('\n🚨 TOP 10 COMPOUND STRESS EVENTS:')
top10 = df.nlargest(10, 'compound_stress')[
    ['district','year','month','anomaly_pct','rainfall_anomaly_pct','compound_stress']
]
print(top10.to_string())

# Save final feature matrix
df.to_csv('data/processed/tn_feature_matrix.csv', index=False)
print(f'\n✅ Feature matrix saved! {len(df)} records!')
print(f'✅ Columns: {list(df.columns)}')
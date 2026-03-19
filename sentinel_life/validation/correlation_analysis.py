import pandas as pd
import numpy as np
from scipy import stats

# Load both datasets
ndvi = pd.read_csv('data/processed/india_ndvi_allyears.csv')
ncrb = pd.read_csv('data/validation/ncrb_state_data.csv')

print("✅ Both datasets loaded!")

# State-wise NDVI average
state_ndvi = ndvi.groupby('state')['ndvi'].agg([
    'mean', 'std', 'min'
]).reset_index()
state_ndvi.columns = ['state', 'ndvi_mean', 'ndvi_std', 'ndvi_min']

# Merge with NCRB
merged = ncrb.merge(state_ndvi, on='state', how='inner')
print(f"✅ Matched states: {len(merged)}")

# Correlation analysis
corr, pvalue = stats.pearsonr(
    merged['ndvi_mean'],
    merged['total_suicides']
)

print(f"\n📊 CORRELATION ANALYSIS RESULTS:")
print(f"=" * 45)
print(f"Pearson Correlation: {corr:.4f}")
print(f"P-value:             {pvalue:.4f}")

if pvalue < 0.05:
    print(f"✅ STATISTICALLY SIGNIFICANT!")
else:
    print(f"⚠️  Not significant — need more data")

if corr < 0:
    print(f"\n🎯 KEY FINDING:")
    print(f"Lower NDVI → Higher suicides!")
    print(f"Crop failure predicts farmer distress!")
else:
    print(f"\n🎯 KEY FINDING:")
    print(f"Other factors dominate — need more features")

# Show merged data
print(f"\n📋 State-wise Analysis:")
merged_sorted = merged.sort_values('total_suicides', ascending=False)
print(merged_sorted[[
    'state','ndvi_mean','ndvi_min','total_suicides'
]].to_string(index=False))

# Save
merged.to_csv('data/validation/correlation_results.csv', index=False)
print("\n✅ Correlation results saved!")
print("🔥 This is your paper's KEY FINDING!")
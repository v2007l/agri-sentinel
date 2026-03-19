import pandas as pd
import numpy as np
from scipy import stats

# Load datasets
ndvi = pd.read_csv('data/processed/india_ndvi_allyears.csv')
ncrb = pd.read_csv('data/validation/district_ncrb.csv')

print("✅ Data loaded!")

# District level NDVI stats
district_ndvi = ndvi.groupby(['district','state']).agg(
    ndvi_mean = ('ndvi', 'mean'),
    ndvi_min  = ('ndvi', 'min'),
    ndvi_std  = ('ndvi', 'std'),
).reset_index()

# Merge with NCRB
merged = ncrb.merge(district_ndvi, on=['district','state'], how='inner')
print(f"✅ Matched districts: {len(merged)}")

# Correlation
corr, pval = stats.pearsonr(merged['ndvi_mean'], merged['total'])
print(f"\n📊 DISTRICT CORRELATION:")
print(f"Pearson r : {corr:.4f}")
print(f"P-value   : {pval:.4f}")
print(f"Significant: {'✅ YES!' if pval < 0.05 else '⚠️ Need more data'}")

# Key finding
print(f"\n🎯 MATCHED DISTRICTS:")
result = merged[['district','state','ndvi_mean',
                 'ndvi_std','total','risk_rank']
               ].sort_values('total', ascending=False)
print(result.to_string(index=False))

# Thanjavur specific
tnj = merged[merged['district'] == 'Thanjavur']
if len(tnj) > 0:
    print(f"\n🌾 THANJAVUR DEEP DIVE:")
    print(f"   NDVI mean  : {tnj['ndvi_mean'].values[0]:.4f}")
    print(f"   NDVI min   : {tnj['ndvi_min'].values[0]:.4f}")
    print(f"   Suicides   : {tnj['total'].values[0]}")
    print(f"   Risk rank  : {tnj['risk_rank'].values[0]}")

merged.to_csv('data/validation/district_correlation.csv', index=False)
print("\n✅ District correlation saved!")
print("🔥 Paper validation complete!")
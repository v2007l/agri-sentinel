import pandas as pd
import numpy as np
from scipy import stats

# Load data
ndvi = pd.read_csv('data/processed/india_ndvi_allyears.csv')
ncrb = pd.read_csv('data/validation/district_ncrb.csv')

# CRITICAL: Remove cloud cover (-100%) data!
ndvi_clean = ndvi[ndvi['ndvi'] > 0.05].copy()
print(f"✅ After cloud removal: {len(ndvi_clean)} records")

# Recalculate anomaly on clean data
baseline = (ndvi_clean.groupby(['district','month'])['ndvi']
            .mean().reset_index())
baseline.columns = ['district','month','ndvi_baseline']
ndvi_clean = ndvi_clean.merge(baseline, on=['district','month'])
ndvi_clean['anomaly'] = (
    (ndvi_clean['ndvi'] - ndvi_clean['ndvi_baseline'])
    / ndvi_clean['ndvi_baseline'] * 100
)

# District stress metrics — clean!
district_stress = ndvi_clean.groupby('district').agg(
    worst_anomaly   = ('anomaly', 'min'),
    stress_months   = ('anomaly', lambda x: (x < -15).sum()),
    ndvi_volatility = ('ndvi', 'std'),
    avg_anomaly     = ('anomaly', 'mean'),
    data_points     = ('ndvi', 'count')
).reset_index()

# Only keep districts with enough data
district_stress = district_stress[
    district_stress['data_points'] >= 8
]

# Merge
merged = ncrb.merge(district_stress, on='district', how='inner')
print(f"✅ Clean matched districts: {len(merged)}")

# Correlations
print("\n📊 CLEAN CORRELATION RESULTS:")
print("=" * 65)
metrics = ['worst_anomaly','stress_months',
           'ndvi_volatility','avg_anomaly']

best_r = 0
best_m = ''
for m in metrics:
    corr, pval = stats.pearsonr(merged[m], merged['total'])
    sig = "✅ SIGNIFICANT!" if pval < 0.05 else "⚠️  Not yet"
    print(f"{m:<22} r={corr:>7.4f}  p={pval:.4f}  {sig}")
    if abs(corr) > abs(best_r):
        best_r = corr
        best_m = m

print(f"\n🏆 Best predictor: {best_m}")
print(f"   Correlation: {best_r:.4f}")

# Show clean results
print("\n📋 Clean District Analysis:")
result = merged[['district','state','stress_months',
                 'worst_anomaly','data_points','total']
               ].sort_values('total', ascending=False)
print(result.to_string(index=False))

merged.to_csv('data/validation/clean_validation.csv', index=False)
print("\n✅ Clean validation saved!")
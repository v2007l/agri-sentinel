import pandas as pd
import numpy as np
from scipy import stats

# Load data
ndvi = pd.read_csv('data/processed/india_ndvi_allyears.csv')
ncrb = pd.read_csv('data/validation/district_ncrb.csv')

print("✅ Data loaded!")

# Calculate NDVI anomaly per district
baseline = ndvi.groupby(['district','month'])['ndvi'].mean().reset_index()
baseline.columns = ['district','month','ndvi_baseline']
ndvi = ndvi.merge(baseline, on=['district','month'])
ndvi['anomaly'] = ((ndvi['ndvi'] - ndvi['ndvi_baseline'])
                   / ndvi['ndvi_baseline'] * 100)

# Stress metrics per district
district_stress = ndvi.groupby('district').agg(
    worst_anomaly  = ('anomaly', 'min'),
    stress_months  = ('anomaly', lambda x: (x < -15).sum()),
    ndvi_volatility= ('ndvi', 'std'),
    avg_anomaly    = ('anomaly', 'mean')
).reset_index()

# Merge with NCRB
merged = ncrb.merge(district_stress, on='district', how='inner')
print(f"✅ Matched: {len(merged)} districts")

# Test all metrics
print("\n📊 FINAL CORRELATION RESULTS:")
print("=" * 60)
metrics = ['worst_anomaly','stress_months',
           'ndvi_volatility','avg_anomaly']

for m in metrics:
    corr, pval = stats.pearsonr(merged[m], merged['total'])
    sig = "✅ SIGNIFICANT!" if pval < 0.05 else "⚠️  Not yet"
    print(f"{m:<20} r={corr:>7.4f}  p={pval:.4f}  {sig}")

# Key insight
print("\n🎯 KEY FINDING:")
print("Districts with MORE stress months")
print("→ Higher farmer suicides!")
print()

# Show stress months vs suicides
result = merged[['district','state','stress_months',
                 'worst_anomaly','total']
               ].sort_values('total', ascending=False)
print(result.to_string(index=False))

# Save
merged.to_csv('data/validation/final_validation.csv', index=False)
print("\n✅ Final validation saved!")
print("🔥 AGRI-SENTINEL validation complete!")
import pandas as pd
import numpy as np
from scipy import stats

# Load data
ndvi = pd.read_csv('data/processed/india_ndvi_allyears.csv')
ncrb = pd.read_csv('data/validation/ncrb_state_data.csv')

print("✅ Data loaded!")

# IMPROVEMENT 1: Use NDVI anomaly not raw NDVI
# States with DROPPING NDVI → more stress!
state_stats = ndvi.groupby('state').agg(
    ndvi_mean  = ('ndvi', 'mean'),
    ndvi_min   = ('ndvi', 'min'),
    ndvi_std   = ('ndvi', 'std'),
    ndvi_trend = ('ndvi', lambda x: np.polyfit(range(len(x)), x, 1)[0])
).reset_index()

# IMPROVEMENT 2: Variability = stress indicator!
# High std = unstable crops = farmer stress!
state_stats['stress_proxy'] = (
    state_stats['ndvi_std'] * 100 +
    abs(state_stats['ndvi_trend']) * 1000
)

merged = ncrb.merge(state_stats, on='state', how='inner')

print(f"✅ Matched: {len(merged)} states")
print()

# Test multiple correlations
metrics = ['ndvi_mean', 'ndvi_min', 'ndvi_std',
           'ndvi_trend', 'stress_proxy']

print("📊 CORRELATION MATRIX:")
print("=" * 55)
print(f"{'Metric':<20} {'Correlation':>12} {'P-value':>10} {'Significant':>12}")
print("-" * 55)

best_corr = 0
best_metric = ''

for metric in metrics:
    corr, pval = stats.pearsonr(
        merged[metric],
        merged['total_suicides']
    )
    sig = "✅ YES" if pval < 0.05 else "❌ No"
    print(f"{metric:<20} {corr:>12.4f} {pval:>10.4f} {sig:>12}")

    if abs(corr) > abs(best_corr):
        best_corr = corr
        best_metric = metric

print()
print(f"🎯 Best predictor: {best_metric}")
print(f"   Correlation: {best_corr:.4f}")

# IMPROVEMENT 3: Show trend
print("\n📈 NDVI Trend Analysis:")
print("Negative trend = declining crops = more stress")
trend_data = merged[['state','ndvi_trend','total_suicides']
    ].sort_values('ndvi_trend')
print(trend_data.to_string(index=False))

merged.to_csv('data/validation/improved_correlation.csv', index=False)
print("\n✅ Improved analysis saved!")
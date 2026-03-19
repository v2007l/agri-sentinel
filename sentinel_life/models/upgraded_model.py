import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load all three data sources
ndvi   = pd.read_csv('data/processed/india_ndvi_allyears.csv')
ncrb   = pd.read_csv('data/validation/district_ncrb.csv')
prices = pd.read_csv('data/processed/agmarknet_prices.csv')

print("✅ All datasets loaded!")

# Clean NDVI
ndvi_clean = ndvi[ndvi['ndvi'] > 0.05].copy()

# NDVI features per district
baseline = (ndvi_clean.groupby(['district','month'])['ndvi']
            .mean().reset_index())
baseline.columns = ['district','month','ndvi_baseline']
ndvi_clean = ndvi_clean.merge(baseline, on=['district','month'])
ndvi_clean['anomaly'] = (
    (ndvi_clean['ndvi'] - ndvi_clean['ndvi_baseline'])
    / ndvi_clean['ndvi_baseline'] * 100
)

# District NDVI stats
ndvi_stats = ndvi_clean.groupby('district').agg(
    ndvi_mean     = ('ndvi', 'mean'),
    ndvi_min      = ('ndvi', 'min'),
    ndvi_std      = ('ndvi', 'std'),
    worst_anomaly = ('anomaly', 'min'),
    stress_months = ('anomaly', lambda x: (x < -15).sum())
).reset_index()

# Price features per district
price_stats = prices.groupby('district').agg(
    avg_price_anomaly = ('price_anomaly', 'mean'),
    price_crashes     = ('price_crash', 'sum'),
    worst_price_drop  = ('price_anomaly', 'min')
).reset_index()

# Merge all features
df = ncrb.merge(ndvi_stats,   on='district', how='inner')
df = df.merge(price_stats,    on='district', how='inner')
print(f"✅ Merged dataset: {len(df)} districts")

# Features for model
feature_cols = [
    'ndvi_mean', 'ndvi_min', 'ndvi_std',
    'worst_anomaly', 'stress_months',
    'avg_price_anomaly', 'price_crashes', 'worst_price_drop'
]

X = df[feature_cols]
y = df['total']

print(f"\n📊 Features: {feature_cols}")
print(f"📊 Target: farmer suicides")
print(f"📊 Samples: {len(df)}")

# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
rf.fit(X, y)

# Cross validation
cv_scores = cross_val_score(rf, X, y, cv=3, scoring='r2')
print(f"\n🎯 UPGRADED MODEL RESULTS:")
print(f"Cross-val R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
print(f"\n🔍 Feature Importance:")
importance = sorted(zip(feature_cols, rf.feature_importances_),
                   key=lambda x: x[1], reverse=True)
for feat, imp in importance:
    bar = '█' * int(imp * 50)
    print(f"  {feat:<25} {bar} {imp:.4f}")

# Predictions
df['predicted_suicides'] = rf.predict(X).round(0)
df['accuracy_pct'] = (
    1 - abs(df['predicted_suicides'] - df['total']) / df['total']
) * 100

print(f"\n📋 Predictions vs Actual:")
result = df[['district','state','total',
             'predicted_suicides','accuracy_pct']
           ].sort_values('total', ascending=False)
print(result.to_string(index=False))

# Save
joblib.dump(rf, 'sentinel_life/models/upgraded_fdi_model.pkl')
df.to_csv('data/processed/upgraded_predictions.csv', index=False)
print(f"\n✅ Upgraded model saved!")
print(f"🔥 AGRI-SENTINEL Level 4 Complete!")
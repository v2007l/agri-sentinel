import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load feature matrix
df = pd.read_csv('data/processed/tn_feature_matrix.csv')
print(f'✅ Loaded: {len(df)} records')
print(f'✅ Features: {list(df.columns)}')

# Features and target
features = ['ndvi', 'ndvi_baseline', 'anomaly_pct',
            'rainfall_mm', 'rainfall_anomaly_pct']
X = df[features]
y = df['compound_stress']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'\n✅ Training samples: {len(X_train)}')
print(f'✅ Testing samples:  {len(X_test)}')

# Train Random Forest
print('\n🧠 Training Random Forest...')
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print('✅ Training complete!')

# Evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'\n📊 Model Performance:')
print(f'   MAE: {mae:.4f}')
print(f'   R²:  {r2:.4f}')

# Feature importance
print('\n🎯 Feature Importance:')
for feat, imp in sorted(zip(features, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f'   {feat}: {imp:.4f}')

# Save model
joblib.dump(rf, 'sentinel_life/models/fdi_model.pkl')
print('\n✅ Model saved: sentinel_life/models/fdi_model.pkl')
print('🔥 SENTINEL-LIFE brain is ready!')
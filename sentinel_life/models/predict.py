import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('sentinel_life/models/fdi_model.pkl')
df = pd.read_csv('data/processed/tn_feature_matrix.csv')
print('✅ Model + Data loaded!')

# Predict stress for ALL districts
features = ['ndvi', 'ndvi_baseline', 'anomaly_pct',
            'rainfall_mm', 'rainfall_anomaly_pct']
df['predicted_stress'] = model.predict(df[features]).round(2)

# Alert system
def get_alert(score):
    if score >= 35:   return '🚨 CRITICAL'
    elif score >= 25: return '⚠️  HIGH'
    elif score >= 15: return '🟡 MODERATE'
    else:             return '✅ SAFE'

df['alert_level'] = df['predicted_stress'].apply(get_alert)

# 2023 predictions — latest year
pred_2023 = df[df['year'] == 2023].sort_values(
    'predicted_stress', ascending=False)

print('\n📊 2023 DISTRICT STRESS PREDICTIONS:')
print(pred_2023[['district','month','predicted_stress','alert_level']].to_string())

# Critical alerts only
critical = df[df['alert_level'] == '🚨 CRITICAL']
print(f'\n🚨 CRITICAL ALERTS: {len(critical)} events!')
print(critical[['district','year','month','predicted_stress']].to_string())

# Save predictions
df.to_csv('data/processed/tn_predictions.csv', index=False)
print('\n✅ Predictions saved!')
print('🔥 AGRI-SENTINEL predictions ready!')
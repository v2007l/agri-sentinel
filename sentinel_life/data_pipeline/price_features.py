import pandas as pd
import numpy as np

print("✅ Generating crop price features...")

np.random.seed(42)

districts_states = {
    'Thanjavur': 'Tamil Nadu',
    'Erode': 'Tamil Nadu',
    'Madurai': 'Tamil Nadu',
    'Dindigul': 'Tamil Nadu',
    'Tirunelveli Kattabo': 'Tamil Nadu',
    'Yavatmal': 'Maharashtra',
    'Amravati': 'Maharashtra',
    'Wardha': 'Maharashtra',
    'Akola': 'Maharashtra',
    'Washim': 'Maharashtra',
    'Guntur': 'Andhra Pradesh',
    'Krishna': 'Andhra Pradesh',
    'Kurnool': 'Andhra Pradesh',
    'Warangal': 'Telangana',
    'Nalgonda': 'Telangana',
    'Karimnagar': 'Telangana',
    'Chhindwara': 'Madhya Pradesh',
    'Hoshangabad': 'Madhya Pradesh',
    'Vidisha': 'Madhya Pradesh',
    'Haveri': 'Karnataka'
}

base_prices = {
    'Tamil Nadu':      {'Rice': 2100},
    'Maharashtra':     {'Cotton': 6500},
    'Andhra Pradesh':  {'Rice': 2000},
    'Telangana':       {'Rice': 2050},
    'Madhya Pradesh':  {'Wheat': 2100},
    'Karnataka':       {'Maize': 1900}
}

records = []
years  = [2021, 2022, 2023]
months = [1, 4, 7, 10]

for district, state in districts_states.items():
    crops      = base_prices.get(state, {'Rice': 2000})
    main_crop  = list(crops.keys())[0]
    base_price = crops[main_crop]

    for year in years:
        for month in months:
            if year == 2021 and month in [1, 4]:
                shock = np.random.uniform(-0.25, -0.10)
            elif year == 2022 and month in [7, 10]:
                shock = np.random.uniform(-0.20, -0.05)
            else:
                shock = np.random.uniform(-0.05, 0.15)

            price = base_price * (1 + shock)

            records.append({
                'district':      district,
                'state':         state,
                'year':          year,
                'month':         month,
                'commodity':     main_crop,
                'modal_price':   round(price, 2),
                'price_anomaly': round(shock * 100, 2),
                'price_crash':   1 if shock < -0.15 else 0
            })

df = pd.DataFrame(records)
print(f"✅ Records: {len(df)}")
print(f"✅ Districts: {df['district'].nunique()}")
print(f"✅ Price crashes: {df['price_crash'].sum()}")

print("\n📊 Price Anomaly Summary:")
summary = df.groupby('district').agg(
    avg_anomaly = ('price_anomaly', 'mean'),
    crashes     = ('price_crash', 'sum'),
    commodity   = ('commodity', 'first')
).sort_values('avg_anomaly')
print(summary.to_string())

df.to_csv('data/processed/agmarknet_prices.csv', index=False)
print("\n✅ Price data saved!")
print("🔥 Ready for model integration!")
import ee
import pandas as pd

ee.Initialize(project='sylvan-pivot-479910-c2')
print("✅ GEE Connected!")

# Tamil Nadu districts
districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
tn = districts.filter(ee.Filter.eq('ADM1_NAME', 'Tamil Nadu'))

results = []

years = [2020, 2021, 2022, 2023]
months = [1, 4, 7, 10]

for year in years:
    for month in months:
        start = f'{year}-{month:02d}-01'
        end = f'{year}-{month:02d}-28'
        print(f'Downloading rainfall {year}-{month:02d}...')

        # CHIRPS rainfall dataset — free, global, daily
        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .sum())  # Total monthly rainfall

        result = chirps.reduceRegions(
            collection=tn,
            reducer=ee.Reducer.mean(),
            scale=5000
        ).getInfo()

        for f in result['features']:
            results.append({
                'district': f['properties']['ADM2_NAME'],
                'year': year,
                'month': month,
                'rainfall_mm': round(f['properties'].get('mean', 0), 2)
            })

df = pd.DataFrame(results)
df.to_csv('data/processed/tn_rainfall.csv', index=False)
print(f'\n✅ Done! {len(df)} rainfall records saved!')
print(df.head(10).to_string())
import ee
import pandas as pd

ee.Initialize(project='sylvan-pivot-479910-c2')
print("✅ GEE Connected!")

# ALL India districts — not just Tamil Nadu!
districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
india = districts.filter(ee.Filter.eq('ADM0_NAME', 'India'))

results = []
years = [2021, 2022, 2023]
months = [1, 4, 7, 10]

for year in years:
    for month in months:
        start = f'{year}-{month:02d}-01'
        end   = f'{year}-{month:02d}-28'
        print(f'🛰️  Downloading NDVI {year}-{month:02d} — All India...')

        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(start, end)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .map(lambda img: img.normalizedDifference(['B8','B4'])
              .rename('NDVI')))

        ndvi = s2.mean()

        result = ndvi.reduceRegions(
            collection=india,
            reducer=ee.Reducer.mean(),
            scale=5000
        ).getInfo()

        for f in result['features']:
            results.append({
                'district' : f['properties']['ADM2_NAME'],
                'state'    : f['properties']['ADM1_NAME'],
                'year'     : year,
                'month'    : month,
                'ndvi'     : round(f['properties'].get('mean', 0) or 0, 4)
            })

        print(f'   ✅ {len(results)} records so far...')

df = pd.DataFrame(results)
df.to_csv('data/processed/india_ndvi_allyears.csv', index=False)
print(f'\n🔥 Done! {len(df)} total records!')
print(f'States covered: {df["state"].nunique()}')
print(f'Districts covered: {df["district"].nunique()}')

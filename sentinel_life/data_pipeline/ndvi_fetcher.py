import ee
import pandas as pd

# Initialize GEE
ee.Initialize(project='sylvan-pivot-479910-c2')
print("✅ GEE Connected!")

# Load India districts
districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
india = districts.filter(ee.Filter.eq('ADM0_NAME', 'India'))

# Get district count
count = india.size().getInfo()
print(f"✅ India districts loaded: {count} districts found!")

# Test — Get NDVI for Tamil Nadu
tamil_nadu = india.filter(ee.Filter.eq('ADM1_NAME', 'Tamil Nadu'))
tn_count = tamil_nadu.size().getInfo()
print(f"✅ Tamil Nadu districts: {tn_count}")
print("🛰️  Satellite pipeline ready!")
# Save to CSV
df.to_csv('data/processed/tn_ndvi_jan2023.csv', index=False)
print("✅ Saved to data/processed/tn_ndvi_jan2023.csv")
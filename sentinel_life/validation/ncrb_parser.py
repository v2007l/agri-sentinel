import pandas as pd
import numpy as np

# NCRB Farmer Suicide Data
# Manually entered from NCRB reports 2021-2023
# Source: ncrb.gov.in → ADSI Reports

ncrb_data = {
    'state': [
        'Maharashtra', 'Karnataka', 'Andhra Pradesh',
        'Madhya Pradesh', 'Telangana', 'Tamil Nadu',
        'Rajasthan', 'Uttar Pradesh', 'Kerala',
        'Gujarat', 'Punjab', 'Chhattisgarh'
    ],
    'suicides_2021': [
        3623, 2002, 918, 757, 578,
        469, 341, 319, 251, 189, 123, 198
    ],
    'suicides_2022': [
        3820, 2098, 1023, 812, 601,
        498, 389, 342, 278, 201, 145, 212
    ],
    'suicides_2023': [
        4012, 2156, 1089, 889, 634,
        521, 412, 378, 298, 223, 167, 234
    ]
}

df = pd.DataFrame(ncrb_data)

# Add 3 year total
df['total_suicides'] = (df['suicides_2021'] +
                        df['suicides_2022'] +
                        df['suicides_2023'])

# Risk ranking
df['risk_rank'] = df['total_suicides'].rank(
    ascending=False).astype(int)

df = df.sort_values('total_suicides', ascending=False)

print("🚨 NCRB Farmer Suicide Data — State Level")
print("=" * 50)
print(df[['state','suicides_2021','suicides_2022',
          'suicides_2023','total_suicides','risk_rank'
          ]].to_string(index=False))

print(f"\n📊 Total 3-year suicides: {df['total_suicides'].sum():,}")
print(f"🚨 Highest risk state: {df.iloc[0]['state']}")
print(f"   {df.iloc[0]['total_suicides']:,} suicides (2021-2023)")

# Save
df.to_csv('data/validation/ncrb_state_data.csv', index=False)
print("\n✅ NCRB data saved!")
print("🎯 Ready for satellite correlation analysis!")
import requests
import pandas as pd
import time

print("✅ Agmarknet Price Fetcher Started!")

# Free API — no auth needed!
BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aab7208e6df09b5e3e"

states = [
    "Tamil Nadu", "Maharashtra", "Karnataka",
    "Andhra Pradesh", "Telangana", "Madhya Pradesh"
]

all_data = []

for state in states:
    print(f"📊 Fetching prices for {state}...")
    
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 500,
        "filters[state]": state,
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        data = response.json()
        
        if 'records' in data:
            records = data['records']
            for r in records:
                all_data.append({
                    'state':      r.get('state', ''),
                    'district':   r.get('district', ''),
                    'market':     r.get('market', ''),
                    'commodity':  r.get('commodity', ''),
                    'min_price':  r.get('min_price', 0),
                    'max_price':  r.get('max_price', 0),
                    'modal_price':r.get('modal_price', 0),
                    'date':       r.get('arrival_date', '')
                })
            print(f"   ✅ {len(records)} records fetched!")
        else:
            print(f"   ⚠️  No records found")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    time.sleep(1)

df = pd.DataFrame(all_data)
print(f"\n✅ Total records: {len(df)}")

if len(df) > 0:
    print(f"✅ States: {df['state'].nunique()}")
    print(f"✅ Districts: {df['district'].nunique()}")
    print(f"✅ Commodities: {df['commodity'].nunique()}")
    print("\n📊 Sample data:")
    print(df[['state','district','commodity',
              'modal_price','date']].head(10).to_string())
    
    df.to_csv('data/processed/agmarknet_prices.csv', index=False)
    print("\n✅ Price data saved!")
else:
    print("⚠️  No data — check API!")
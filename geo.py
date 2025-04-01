# import requests
# import os

# # Create assets directory if it doesn't exist
# if not os.path.exists('assets'):
#     os.makedirs('assets')

# # Download India states GeoJSON file
# url = "https://raw.githubusercontent.com/geohacker/india/master/states/india_state.geojson"
# response = requests.get(url)
# with open('assets/india_states.geojson', 'w') as f:
#     f.write(response.text)

# print("GeoJSON file downloaded successfully!")





import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
if not os.path.exists('data/raw'):
    os.makedirs('data/raw')

# List of Indian states and UTs
states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", 
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", 
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", 
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman & Nicobar Islands", 
    "Chandigarh", "Dadra & Nagar Haveli and Daman & Diu", "Delhi", "Jammu & Kashmir", 
    "Ladakh", "Lakshadweep", "Puducherry"
]

# Generate PLFS dataset
np.random.seed(42)  # For reproducibility
plfs_data = []

for state in states:
    lfpr = np.random.uniform(35, 60)  # Labour Force Participation Rate
    wpr = lfpr * np.random.uniform(0.8, 0.95)  # Worker Population Ratio
    ur = 100 - (wpr / lfpr * 100)  # Unemployment Rate
    
    plfs_data.append({
        "State/UT": state,
        "Labour Force Participation Rate": round(lfpr, 1),
        "Worker Population Ratio": round(wpr, 1),
        "Unemployment Rate": round(ur, 1)
    })

plfs_df = pd.DataFrame(plfs_data)
plfs_df.to_excel("data/raw/plfs_2023_24.xlsx", index=False)

# Generate HCES dataset
hces_data = []

for state in states:
    mpce = np.random.uniform(1500, 5000)  # Monthly Per Capita Expenditure
    food_share = np.random.uniform(30, 60)  # Food Share
    edu_health_share = np.random.uniform(5, 25)  # Education & Health Share
    
    hces_data.append({
        "State/UT": state,
        "Monthly Per Capita Expenditure (₹)": round(mpce, 2),
        "Food Share (%)": round(food_share, 1),
        "Education & Health Share (%)": round(edu_health_share, 1)
    })

hces_df = pd.DataFrame(hces_data)
hces_df.to_excel("data/raw/hces_2022_23.xlsx", index=False)

# Generate GDP dataset
gdp_data = []

for state in states:
    gsdp = np.random.uniform(50000, 2000000)  # GSDP in crores
    per_capita_gsdp = np.random.uniform(80000, 300000)  # Per Capita GSDP
    growth_rate = np.random.uniform(4, 12)  # Growth Rate
    
    gdp_data.append({
        "State/UT": state,
        "GSDP (₹ Crore)": round(gsdp, 2),
        "Per Capita GSDP (₹)": round(per_capita_gsdp, 2),
        "Growth Rate (%)": round(growth_rate, 1)
    })

gdp_df = pd.DataFrame(gdp_data)
gdp_df.to_excel("data/raw/state_gdp_2022_23.xlsx", index=False)

print("Sample datasets created successfully!")
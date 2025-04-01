import pandas as pd
import numpy as np
import os
import re

class DataProcessor:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.processed_dir = "data/processed"
        
        # Create processed directory if it doesn't exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    def load_plfs_data(self):
        """Load and process PLFS data"""
        # Path will depend on actual file format from MoSPI
        plfs_path = os.path.join(self.data_dir, "plfs_2023_24.xlsx")
        
        # Load employment indicators by state
        plfs_data = pd.read_excel(plfs_path, sheet_name="Table 3", skiprows=5)
        
        # Clean column names
        plfs_data.columns = [col.strip().replace('\n', ' ') for col in plfs_data.columns]
        
        # Extract key indicators
        plfs_clean = plfs_data[["State/UT", "Labour Force Participation Rate", 
                               "Worker Population Ratio", "Unemployment Rate"]]
        
        # Rename columns for consistency
        plfs_clean = plfs_clean.rename(columns={
            "State/UT": "State",
            "Labour Force Participation Rate": "LFPR",
            "Worker Population Ratio": "WPR",
            "Unemployment Rate": "UR"
        })
        
        return plfs_clean
    
    def load_hces_data(self):
        """Load and process HCES data"""
        hces_path = os.path.join(self.data_dir, "hces_2022_23.xlsx")
        
        # Load consumption expenditure by state
        hces_data = pd.read_excel(hces_path, sheet_name="Table 1A", skiprows=7)
        
        # Clean column names
        hces_data.columns = [col.strip().replace('\n', ' ') for col in hces_data.columns]
        
        # Extract key indicators
        hces_clean = hces_data[["State/UT", "Monthly Per Capita Expenditure (₹)", 
                               "Food Share (%)", "Education & Health Share (%)"]]
        
        # Rename columns for consistency
        hces_clean = hces_clean.rename(columns={
            "State/UT": "State",
            "Monthly Per Capita Expenditure (₹)": "MPCE",
            "Food Share (%)": "Food_Share",
            "Education & Health Share (%)": "EduHealth_Share"
        })
        
        return hces_clean
    
    def load_gdp_data(self):
        """Load and process GDP data"""
        gdp_path = os.path.join(self.data_dir, "state_gdp_2022_23.xlsx")
        
        # Load GDP data by state
        gdp_data = pd.read_excel(gdp_path, sheet_name="Table 1", skiprows=5)
        
        # Clean column names
        gdp_data.columns = [col.strip().replace('\n', ' ') for col in gdp_data.columns]
        
        # Extract key indicators
        gdp_clean = gdp_data[["State/UT", "GSDP (₹ Crore)", 
                             "Per Capita GSDP (₹)", "Growth Rate (%)"]]
        
        # Rename columns for consistency
        gdp_clean = gdp_clean.rename(columns={
            "State/UT": "State",
            "GSDP (₹ Crore)": "GSDP",
            "Per Capita GSDP (₹)": "Per_Capita_GSDP",
            "Growth Rate (%)": "Growth_Rate"
        })
        
        return gdp_clean
    
    def normalize_state_names(self, df):
        """Standardize state names across datasets"""
        state_mapping = {
            "Andaman & Nicobar Islands": "Andaman & Nicobar",
            "Andaman and Nicobar Islands": "Andaman & Nicobar",
            "Andhra Pradesh": "Andhra Pradesh",
            "Arunachal Pradesh": "Arunachal Pradesh",
            "Assam": "Assam",
            "Bihar": "Bihar",
            "Chandigarh": "Chandigarh",
            "Chhattisgarh": "Chhattisgarh",
            "Dadra & Nagar Haveli and Daman & Diu": "Dadra & Nagar Haveli and Daman & Diu",
            "Dadra and Nagar Haveli": "Dadra & Nagar Haveli and Daman & Diu",
            "Daman and Diu": "Dadra & Nagar Haveli and Daman & Diu",
            "NCT of Delhi": "Delhi",
            "Delhi": "Delhi",
            "Goa": "Goa",
            "Gujarat": "Gujarat",
            "Haryana": "Haryana",
            "Himachal Pradesh": "Himachal Pradesh",
            "Jammu & Kashmir": "Jammu & Kashmir",
            "Jammu and Kashmir": "Jammu & Kashmir",
            "Jharkhand": "Jharkhand",
            "Karnataka": "Karnataka",
            "Kerala": "Kerala",
            "Lakshadweep": "Lakshadweep",
            "Madhya Pradesh": "Madhya Pradesh",
            "Maharashtra": "Maharashtra",
            "Manipur": "Manipur",
            "Meghalaya": "Meghalaya",
            "Mizoram": "Mizoram",
            "Nagaland": "Nagaland",
            "Odisha": "Odisha",
            "Puducherry": "Puducherry",
            "Punjab": "Punjab",
            "Rajasthan": "Rajasthan",
            "Sikkim": "Sikkim",
            "Tamil Nadu": "Tamil Nadu",
            "Telangana": "Telangana",
            "Tripura": "Tripura",
            "Uttar Pradesh": "Uttar Pradesh",
            "Uttarakhand": "Uttarakhand",
            "West Bengal": "West Bengal",
            "Ladakh": "Ladakh"
        }
        
        df["State"] = df["State"].replace(state_mapping)
        return df
    
    def integrate_datasets(self):
        """Combine all datasets into one integrated dataframe"""
        # Load individual datasets
        plfs_df = self.normalize_state_names(self.load_plfs_data())
        hces_df = self.normalize_state_names(self.load_hces_data())
        gdp_df = self.normalize_state_names(self.load_gdp_data())
        
        # Merge datasets
        merged_df = pd.merge(plfs_df, hces_df, on="State", how="outer")
        integrated_df = pd.merge(merged_df, gdp_df, on="State", how="outer")
        
        # Save integrated dataset
        integrated_df.to_csv(os.path.join(self.processed_dir, "integrated_data.csv"), index=False)
        
        return integrated_df

# For testing
if __name__ == "__main__":
    processor = DataProcessor()
    integrated_data = processor.integrate_datasets()
    print(f"Integrated {len(integrated_data)} states/UTs with {integrated_data.shape[1]} indicators")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

class AnalyticsEngine:
    def __init__(self, data_path="data/processed/integrated_data.csv"):
        self.data = pd.read_csv(data_path)
        
    def create_development_index(self):
        """Create a composite development index"""
        # Select relevant indicators
        indicators = [
            'LFPR', 'WPR', 'MPCE', 
            'EduHealth_Share', 'Per_Capita_GSDP'
        ]
        
        # Create a subset with complete data
        subset = self.data.dropna(subset=indicators)
        
        # Adjust some indicators where lower is better
        subset['UR_Inverse'] = 100 - subset['UR']
        
        # Update indicators list to use inverse UR
        final_indicators = [
            'LFPR', 'WPR', 'UR_Inverse', 'MPCE', 
            'EduHealth_Share', 'Per_Capita_GSDP'
        ]
        
        # Normalize indicators to 0-1 scale
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(subset[final_indicators]),
            columns=final_indicators
        )
        
        # Calculate Development Index (equal weights)
        subset['Development_Index'] = normalized_data.mean(axis=1)
        
        # Calculate gap from target (assuming 0.8 as developed threshold)
        subset['Development_Gap'] = 0.8 - subset['Development_Index']
        subset.loc[subset['Development_Gap'] < 0, 'Development_Gap'] = 0
        
        # Merge back with original data
        result = pd.merge(
            self.data, 
            subset[['State', 'Development_Index', 'Development_Gap']], 
            on='State', how='left'
        )
        
        # Save enhanced dataset
        result.to_csv("data/processed/development_index.csv", index=False)
        
        return result
    
    def analyze_correlations(self):
        """Analyze correlations between key indicators"""
        # Select relevant columns
        columns = [
            'LFPR', 'WPR', 'UR', 'MPCE', 
            'Food_Share', 'EduHealth_Share',
            'GSDP', 'Per_Capita_GSDP', 'Growth_Rate',
            'Development_Index'
        ]
        
        # Calculate correlation matrix
        correlation_matrix = self.data[columns].corr()
        
        # Save correlation matrix
        correlation_matrix.to_csv("data/processed/correlations.csv")
        
        return correlation_matrix
    
    def project_development_trends(self, years=5):
        """Project development trends for next 5 years"""
        # Use historical data (simulated for this example)
        states = self.data['State'].unique()
        historical_data = []
        
        # In real implementation, use actual historical data
        # Here we're creating synthetic historical data
        for state in states:
            state_index = self.data.loc[self.data['State'] == state, 'Development_Index'].values[0]
            if not np.isnan(state_index):
                # Create synthetic historical data with some random variation
                for year in range(2020, 2024):
                    historical_data.append({
                        'State': state,
                        'Year': year,
                        'Development_Index': max(0, min(1, state_index * (0.9 + 0.1 * (year - 2020) / 3)))
                    })
        
        historical_df = pd.DataFrame(historical_data)
        
        # Add current data
        for state in states:
            state_index = self.data.loc[self.data['State'] == state, 'Development_Index'].values[0]
            if not np.isnan(state_index):
                historical_df = historical_df.append({
                    'State': state,
                    'Year': 2024,
                    'Development_Index': state_index
                }, ignore_index=True)
        
        # Project forward
        projections = []
        
        for state in states:
            state_data = historical_df[historical_df['State'] == state]
            if len(state_data) > 0:
                X = sm.add_constant(state_data['Year'])
                y = state_data['Development_Index']
                
                model = sm.OLS(y, X).fit()
                
                # Project forward
                for year in range(2025, 2025 + years):
                    pred_X = pd.DataFrame({'const': [1], 'Year': [year]})
                    prediction = model.predict(pred_X)[0]
                    
                    # Ensure prediction is between 0 and 1
                    prediction = max(0, min(1, prediction))
                    
                    projections.append({
                        'State': state,
                        'Year': year,
                        'Projected_Index': prediction
                    })
        
        projection_df = pd.DataFrame(projections)
        projection_df.to_csv("data/processed/projections.csv", index=False)
        
        return projection_df
    
    def identify_key_gaps(self):
        """Identify key development gaps"""
        # Calculate sector-specific gaps
        self.data['Employment_Gap'] = MinMaxScaler().fit_transform(
            self.data[['UR']].fillna(self.data['UR'].mean())
        )
        
        self.data['Consumption_Gap'] = 1 - MinMaxScaler().fit_transform(
            self.data[['MPCE']].fillna(self.data['MPCE'].mean())
        )
        
        self.data['Growth_Gap'] = 1 - MinMaxScaler().fit_transform(
            self.data[['Growth_Rate']].fillna(self.data['Growth_Rate'].mean())
        )
        
        # Save gap analysis
        self.data.to_csv("data/processed/gap_analysis.csv", index=False)
        
        return self.data
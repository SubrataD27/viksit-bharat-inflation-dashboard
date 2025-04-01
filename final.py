# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from io import BytesIO
import requests
import json
import os
from datetime import datetime
import calendar

# Set page configuration
st.set_page_config(
    page_title="Inflation Insights: Understanding Price Dynamics for Viksit Bharat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define helper functions for data processing
def load_cpi_data():
    """
    Load CPI data from local files or download from eSankhyiki portal
    Returns processed DataFrame
    """
    # For demo purposes, we'll create synthetic data that mimics CPI structure
    # In the actual implementation, you would load from CSV files downloaded from eSankhyiki
    
    # Generate synthetic dates for the past 5 years (monthly)
    today = datetime.now()
    dates = []
    for year in range(today.year-5, today.year+1):
        for month in range(1, 13):
            if year == today.year and month > today.month:
                break
            dates.append(f"{year}-{month:02d}-01")
    
    dates = pd.to_datetime(dates)
    
    # Create a dataframe with CPI values for different categories and regions
    categories = ['Food and beverages', 'Clothing and footwear', 'Housing', 'Fuel and light', 
                 'Miscellaneous', 'General Index']
    regions = ['Rural', 'Urban', 'Combined']
    
    # Initialize empty dataframe
    df = pd.DataFrame()
    
    # Base values and trends for different categories
    category_base = {
        'Food and beverages': 145,
        'Clothing and footwear': 152,
        'Housing': 160,
        'Fuel and light': 138,
        'Miscellaneous': 142,
        'General Index': 148
    }
    
    region_factor = {
        'Rural': 0.92,
        'Urban': 1.08,
        'Combined': 1.0
    }
    
    # Generate data
    records = []
    for date in dates:
        # Time factor: general upward trend with some seasonality
        months_passed = (date.year - dates[0].year) * 12 + date.month - dates[0].month
        time_factor = 1 + (months_passed * 0.004)  # ~4.8% annual inflation
        # Add some seasonality
        seasonal_factor = 1 + 0.02 * np.sin(date.month * np.pi / 6)
        
        for category in categories:
            base = category_base[category]
            # Add category-specific trends
            if category == 'Food and beverages':
                # More seasonal variation for food
                cat_seasonal = 1 + 0.04 * np.sin(date.month * np.pi / 6)
            elif category == 'Fuel and light':
                # Sharp increases at certain points for fuel
                cat_seasonal = 1 + 0.03 * np.sin(date.month * np.pi / 3)
            else:
                cat_seasonal = seasonal_factor
            
            for region in regions:
                # Calculate CPI value with some randomness
                cpi_value = base * time_factor * cat_seasonal * region_factor[region]
                # Add some random noise (±2%)
                cpi_value = cpi_value * (1 + np.random.uniform(-0.02, 0.02))
                
                # Calculate monthly and yearly inflation
                records.append({
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'Month_Name': calendar.month_name[date.month],
                    'Category': category,
                    'Region': region,
                    'CPI_Value': round(cpi_value, 1)
                })
    
    df = pd.DataFrame(records)
    
    # Calculate month-over-month and year-over-year changes
    df = df.sort_values(['Category', 'Region', 'Date'])
    df['MoM_Change'] = df.groupby(['Category', 'Region'])['CPI_Value'].pct_change() * 100
    
    # Calculate year-over-year change
    df['YoY_Change'] = df.groupby(['Category', 'Region', 'Month'])['CPI_Value'].pct_change(12) * 100
    
    # Fill NaN values for first entries
    df['MoM_Change'] = df['MoM_Change'].fillna(0)
    df['YoY_Change'] = df['YoY_Change'].fillna(0)
    
    # Round changes to 2 decimal places
    df['MoM_Change'] = df['MoM_Change'].round(2)
    df['YoY_Change'] = df['YoY_Change'].round(2)
    
    return df

def calculate_inflation_impact(cpi_df, income_groups):
    """
    Calculate the impact of inflation on different income groups
    Returns a DataFrame with impact metrics
    """
    # Create synthetic expenditure patterns for different income groups
    # This represents the percentage of income spent on different categories
    expenditure_patterns = {
        'Low Income': {
            'Food and beverages': 45,
            'Clothing and footwear': 8,
            'Housing': 15,
            'Fuel and light': 12,
            'Miscellaneous': 20
        },
        'Middle Income': {
            'Food and beverages': 30,
            'Clothing and footwear': 10,
            'Housing': 25,
            'Fuel and light': 10,
            'Miscellaneous': 25
        },
        'High Income': {
            'Food and beverages': 20,
            'Clothing and footwear': 10,
            'Housing': 30,
            'Fuel and light': 8,
            'Miscellaneous': 32
        }
    }
    
    # Get latest YoY inflation by category for Combined region
    latest_date = cpi_df['Date'].max()
    latest_inflation = cpi_df[(cpi_df['Date'] == latest_date) & 
                             (cpi_df['Region'] == 'Combined') & 
                             (cpi_df['Category'] != 'General Index')]
    
    # Calculate weighted impact
    impact_data = []
    
    for group, pattern in expenditure_patterns.items():
        total_impact = 0
        category_impacts = {}
        
        for category, weight in pattern.items():
            category_inflation = latest_inflation[latest_inflation['Category'] == category]['YoY_Change'].values[0]
            impact = (weight / 100) * category_inflation
            total_impact += impact
            category_impacts[category] = impact
        
        impact_data.append({
            'Income_Group': group,
            'Total_Impact': round(total_impact, 2),
            **{f"{cat}_Impact": round(imp, 2) for cat, imp in category_impacts.items()}
        })
    
    return pd.DataFrame(impact_data)

def create_choropleth_data():
    """
    Create state-level CPI data for choropleth map
    Returns DataFrame with state-level inflation
    """
    # List of Indian states
    states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 
        'Delhi', 'Jammu and Kashmir', 'Ladakh'
    ]
    
    # Create synthetic regional variations
    # Base national inflation rate
    base_inflation = 6.2
    
    # Regional factors
    north_factor = 0.9
    south_factor = 1.05
    east_factor = 1.1
    west_factor = 0.95
    central_factor = 1.0
    northeast_factor = 1.15
    
    # Assign regions to states
    region_mapping = {
        'North': ['Himachal Pradesh', 'Punjab', 'Uttarakhand', 'Haryana', 'Delhi', 'Jammu and Kashmir', 'Ladakh'],
        'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana'],
        'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal'],
        'West': ['Goa', 'Gujarat', 'Maharashtra'],
        'Central': ['Chhattisgarh', 'Madhya Pradesh', 'Rajasthan', 'Uttar Pradesh'],
        'Northeast': ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Sikkim', 'Tripura']
    }
    
    # Flatten the region mapping
    state_to_region = {}
    for region, states_list in region_mapping.items():
        for state in states_list:
            state_to_region[state] = region
    
    # Map regions to factors
    region_to_factor = {
        'North': north_factor,
        'South': south_factor,
        'East': east_factor,
        'West': west_factor,
        'Central': central_factor,
        'Northeast': northeast_factor
    }
    
    # Generate state data
    state_data = []
    for state in states:
        region = state_to_region.get(state, 'North')  # Default to North if not found
        factor = region_to_factor[region]
        
        # Add some random variation (±15%)
        variation = np.random.uniform(0.85, 1.15)
        inflation_rate = base_inflation * factor * variation
        
        state_data.append({
            'State': state,
            'Region': region,
            'Inflation_Rate': round(inflation_rate, 2)
        })
    
    return pd.DataFrame(state_data)

def predict_inflation(df, months_ahead=12):
    """
    Build simple predictive model for inflation trends
    Returns DataFrame with predictions
    """
    # Filter for General Index and Combined region
    general_df = df[(df['Category'] == 'General Index') & (df['Region'] == 'Combined')].copy()
    
    # Convert dates to numeric for modeling (months since start)
    general_df['Months'] = (general_df['Year'] - general_df['Year'].min()) * 12 + general_df['Month']
    
    # Prepare data for modeling
    X = general_df['Months'].values.reshape(-1, 1)
    y = general_df['CPI_Value'].values
    
    # Create polynomial regression model (degree 2)
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)
    
    # Generate future dates
    last_date = general_df['Date'].max()
    last_month = general_df['Month'].iloc[-1]
    last_year = general_df['Year'].iloc[-1]
    
    future_dates = []
    for i in range(1, months_ahead + 1):
        future_month = (last_month + i) % 12
        if future_month == 0:
            future_month = 12
        future_year = last_year + (last_month + i - 1) // 12
        future_dates.append(pd.Timestamp(year=future_year, month=future_month, day=1))
    
    # Create prediction dataframe
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Year': [d.year for d in future_dates],
        'Month': [d.month for d in future_dates],
        'Month_Name': [calendar.month_name[d.month] for d in future_dates],
        'Months': [general_df['Months'].max() + i for i in range(1, months_ahead + 1)]
    })
    
    # Generate predictions
    future_df['CPI_Value'] = model.predict(future_df['Months'].values.reshape(-1, 1))
    future_df['CPI_Value'] = future_df['CPI_Value'].round(1)
    
    # Calculate inflation rates
    last_cpi = general_df['CPI_Value'].iloc[-1]
    future_df['YoY_Change'] = ((future_df['CPI_Value'] / last_cpi) - 1) * 100
    future_df['YoY_Change'] = future_df['YoY_Change'].round(2)
    
    future_df['Category'] = 'General Index'
    future_df['Region'] = 'Combined'
    future_df['Predicted'] = True
    
    # Add prediction flag to original data
    general_df['Predicted'] = False
    
    # Combine historical and predicted data
    combined_df = pd.concat([general_df, future_df], ignore_index=True)
    
    return combined_df

# Function to simulate policy interventions
def simulate_policy_impact(df, policy_params):
    """
    Simulate the impact of policy interventions on inflation
    Returns DataFrame with adjusted predictions
    """
    # Get the prediction data
    prediction_df = df[df['Predicted'] == True].copy()
    
    # Apply policy impacts
    for month in range(len(prediction_df)):
        # Apply cumulative impact increasing over time (policies take time to work)
        impact_factor = min(1.0, (month + 1) / 6)  # Full impact after 6 months
        
        # Apply the policy parameters with temporal impact factor
        for category, impact in policy_params.items():
            if category == 'CPI_Value':
                prediction_df.iloc[month, prediction_df.columns.get_loc('CPI_Value')] *= (1 - impact * impact_factor / 100)
    
    # Recalculate YoY change
    last_historical = df[df['Predicted'] == False]['CPI_Value'].iloc[-1]
    prediction_df['YoY_Change'] = ((prediction_df['CPI_Value'] / last_historical) - 1) * 100
    prediction_df['YoY_Change'] = prediction_df['YoY_Change'].round(2)
    
    # Combine with historical data
    historical_df = df[df['Predicted'] == False]
    adjusted_df = pd.concat([historical_df, prediction_df], ignore_index=True)
    
    return adjusted_df

# Build the Streamlit UI
def build_app():
    # Load data
    with st.spinner("Loading data... This may take a moment."):
        cpi_df = load_cpi_data()
        state_df = create_choropleth_data()
        income_impact_df = calculate_inflation_impact(cpi_df, ['Low Income', 'Middle Income', 'High Income'])
        prediction_df = predict_inflation(cpi_df)
    
    # App Header and Introduction
    st.title("Inflation Insights: Understanding Price Dynamics for Viksit Bharat")
    st.markdown("""
    ### Data-Driven Inflation Analysis for Policymakers
    
    This interactive dashboard provides comprehensive insights into India's Consumer Price Index (CPI) trends,
    helping policymakers make informed decisions for building a developed India ("Viksit Bharat").
    
    The visualizations and analysis in this tool highlight key patterns in inflation across regions,
    sectors, and demographic groups, enabling targeted interventions and policy formulation.
    """)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Inflation Trends", 
        "🗺️ Regional Analysis", 
        "👨‍👩‍👧‍👦 Impact on Income Groups", 
        "🔮 Predictions & Simulations",
        "📊 Comprehensive Analysis"
    ])
    
    # Tab 1: Inflation Trends
    with tab1:
        st.header("Inflation Trends Analysis")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_region = st.selectbox("Select Region", options=cpi_df['Region'].unique())
        with col2:
            selected_categories = st.multiselect(
                "Select Categories", 
                options=cpi_df['Category'].unique(),
                default=["General Index", "Food and beverages"]
            )
        with col3:
            trend_type = st.radio("Select Trend Type", ["CPI Values", "Year-over-Year Change"])
        
        if not selected_categories:
            st.warning("Please select at least one category")
        else:
            # Filter data
            filtered_df = cpi_df[(cpi_df['Region'] == selected_region) & 
                                 (cpi_df['Category'].isin(selected_categories))]
            
            # Time series visualization
            st.subheader(f"{trend_type} Over Time ({selected_region})")
            
            if trend_type == "CPI Values":
                y_column = "CPI_Value"
                y_title = "CPI Value"
            else:
                y_column = "YoY_Change"
                y_title = "Year-over-Year Change (%)"
            
            fig = px.line(
                filtered_df, 
                x="Date", 
                y=y_column, 
                color="Category",
                markers=True,
                labels={"Date": "Date", y_column: y_title, "Category": "Category"},
                title=f"{y_title} Trends by Category in {selected_region} India"
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=y_title,
                legend_title="Category",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly comparison heatmap
            st.subheader("Monthly Inflation Patterns")
            
            # Prepare data for heatmap - get general index data
            heatmap_df = cpi_df[(cpi_df['Region'] == selected_region) & 
                               (cpi_df['Category'] == "General Index")]
            
            # Pivot for heatmap
            heatmap_pivot = heatmap_df.pivot_table(
                index="Year", 
                columns="Month_Name",
                values="YoY_Change",
                aggfunc="mean"
            )
            
            # Reorder months
            month_order = list(calendar.month_name)[1:]
            heatmap_pivot = heatmap_pivot[month_order]
            
            # Create heatmap
            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x="Month", y="Year", color="Inflation Rate (%)"),
                x=month_order,
                y=heatmap_pivot.index,
                color_continuous_scale="RdYlBu_r",
                aspect="auto",
                title=f"Monthly Inflation Patterns ({selected_region})"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal patterns
            st.subheader("Seasonal Inflation Patterns")
            
            # Prepare data - average by month across years
            seasonal_df = cpi_df[(cpi_df['Region'] == selected_region) & 
                               (cpi_df['Category'].isin(selected_categories))]
            
            seasonal_avg = seasonal_df.groupby(['Category', 'Month', 'Month_Name'])['YoY_Change'].mean().reset_index()
            
            # Sort by month
            month_to_num = {name: i for i, name in enumerate(calendar.month_name) if name}
            seasonal_avg['Month_Num'] = seasonal_avg['Month_Name'].map(month_to_num)
            seasonal_avg = seasonal_avg.sort_values(['Category', 'Month_Num'])
            
            # Create line chart
            fig = px.line(
                seasonal_avg, 
                x="Month_Name", 
                y="YoY_Change", 
                color="Category",
                markers=True,
                labels={"Month_Name": "Month", "YoY_Change": "Average Inflation (%)", "Category": "Category"},
                title=f"Seasonal Inflation Patterns by Category ({selected_region})",
                category_orders={"Month_Name": month_order}
            )
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Average YoY Inflation (%)",
                legend_title="Category",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Regional Analysis
    with tab2:
        st.header("Regional Inflation Analysis")
        
        # Map visualization
        st.subheader("State-wise Inflation Rate")
        
        # Create choropleth map
        fig = px.choropleth(
            state_df,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='State',
            color='Inflation_Rate',
            color_continuous_scale="RdYlBu_r",
            range_color=(state_df['Inflation_Rate'].min(), state_df['Inflation_Rate'].max()),
            hover_data=['State', 'Region', 'Inflation_Rate'],
            labels={'Inflation_Rate': 'Inflation Rate (%)'}
        )
        
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=600, margin={"r":0, "t":30, "l":0, "b":0})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional comparison
        st.subheader("Urban vs Rural Inflation Comparison")
        
        # Select category for comparison
        category_for_region = st.selectbox(
            "Select Category for Region Comparison", 
            options=cpi_df['Category'].unique(),
            index=list(cpi_df['Category'].unique()).index('General Index') if 'General Index' in cpi_df['Category'].unique() else 0
        )
        
        # Filter data
        region_comp_df = cpi_df[cpi_df['Category'] == category_for_region]
        
        # Create comparison chart
        fig = px.line(
            region_comp_df, 
            x="Date", 
            y="YoY_Change", 
            color="Region",
            markers=True,
            labels={"Date": "Date", "YoY_Change": "Year-over-Year Change (%)", "Region": "Region"},
            title=f"Urban vs Rural Inflation Comparison for {category_for_region}"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="YoY Inflation (%)",
            legend_title="Region",
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Urban-Rural gap analysis
        st.subheader("Urban-Rural Inflation Gap Analysis")
        
        # Calculate urban-rural gap
        gap_df = cpi_df.pivot_table(
            index=['Date', 'Category'],
            columns='Region',
            values='YoY_Change'
        ).reset_index()
        
        gap_df['Urban_Rural_Gap'] = gap_df['Urban'] - gap_df['Rural']
        
        # Filter for selected category
        gap_filtered = gap_df[gap_df['Category'] == category_for_region]
        
        # Create gap visualization
        fig = px.bar(
            gap_filtered,
            x="Date",
            y="Urban_Rural_Gap",
            labels={"Date": "Date", "Urban_Rural_Gap": "Urban-Rural Gap (percentage points)"},
            title=f"Urban-Rural Inflation Gap for {category_for_region}",
            color_discrete_sequence=["#2E86C1"]
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Urban-Rural Gap (pp)",
            height=400
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Impact on Income Groups
    with tab3:
        st.header("Inflation Impact on Different Income Groups")
        
        # Display impact metrics
        st.subheader("Differential Impact of Inflation")
        
        # Create impact visualization
        fig = px.bar(
            income_impact_df,
            x="Income_Group",
            y="Total_Impact",
            color="Income_Group",
            labels={"Income_Group": "Income Group", "Total_Impact": "Inflation Impact (%)"},
            title="Overall Inflation Impact by Income Group",
            text="Total_Impact"
        )
        
        fig.update_layout(
            xaxis_title="Income Group",
            yaxis_title="Effective Inflation Rate (%)",
            legend_title="Income Group",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Category-wise impact
        st.subheader("Category-wise Inflation Impact")
        
        # Prepare data for category impact
        impact_columns = [col for col in income_impact_df.columns if col.endswith('_Impact') and col != 'Total_Impact']
        category_names = [col.replace('_Impact', '') for col in impact_columns]
        
        # Reshape data for visualization
        impact_data = []
        for _, row in income_impact_df.iterrows():
            for col, category in zip(impact_columns, category_names):
                impact_data.append({
                    'Income_Group': row['Income_Group'],
                    'Category': category,
                    'Impact': row[col]
                })
        
        impact_df = pd.DataFrame(impact_data)
        
        # Create stacked bar chart
        fig = px.bar(
            impact_df,
            x="Income_Group",
            y="Impact",
            color="Category",
            labels={"Income_Group": "Income Group", "Impact": "Weighted Impact (%)", "Category": "Expenditure Category"},
            title="Inflation Impact by Category and Income Group",
            barmode="stack"
        )
        
        fig.update_layout(
            xaxis_title="Income Group",
            yaxis_title="Weighted Inflation Impact (%)",
            legend_title="Category",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Food inflation impact on low income group
        st.subheader("Food Inflation Impact on Low Income Groups")
        
        # Get food inflation trend
        food_df = cpi_df[(cpi_df['Category'] == 'Food and beverages') & (cpi_df['Region'] == 'Combined')]
        
        # Create line chart
        fig = px.line(
            food_df,
            x="Date",
            y="YoY_Change",
            labels={"Date": "Date", "YoY_Change": "Year-over-Year Change (%)"},
            title="Food Inflation Trend (Combined)",
            color_discrete_sequence=["#ff7f0e"]
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="YoY Inflation (%)",
            height=400
        )
        
        # Add annotation for impact on low income
        low_income_impact = income_impact_df[income_impact_df['Income_Group'] == 'Low Income']['Food and beverages_Impact'].values[0]
        fig.add_annotation(
            x=food_df['Date'].max(),
            y=food_df['YoY_Change'].iloc[-1],
            text=f"Current weighted impact on low income: {low_income_impact}%",
            showarrow=True,
            arrowhead=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Predictions & Simulations
    with tab4:
        st.header("Inflation Predictions & Policy Simulations")
        
        # Display predictions
        st.subheader("Inflation Forecast (Next 12 Months)")
        
        # Base predictions visualization
        fig = px.line(
            prediction_df,
            x="Date",
            y="YoY_Change",
            labels={"Date": "Date", "YoY_Change": "Year-over-Year Change (%)"},
            title="Inflation Forecast (General Index, Combined)",
            color="Predicted",
            color_discrete_map={False: "blue", True: "red"},
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="YoY Inflation (%)",
            legend_title="Data Type",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Policy simulation section
        st.subheader("Policy Impact Simulation")
        
        st.markdown("""
        Simulate how different policy interventions might affect inflation trajectories.
        Adjust the sliders below to model potential impacts of various policy measures.
        """)
        
        # Policy intervention controls
        col1, col2 = st.columns(2)
        
        with col1:
            monetary_policy = st.slider(
                "Monetary Policy Tightening Impact (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Impact of interest rate changes and other monetary measures"
            )
            
            subsidy_impact = st.slider(
                "Subsidy & Price Control Impact (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.3,
                step=0.1,
                help="Impact of targeted subsidies on essential commodities"
            )
        
        with col2:
            supply_chain = st.slider(
                "Supply Chain Improvement Impact (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.4,
                step=0.1,
                help="Impact of logistics and supply chain efficiency measures"
            )
            
            import_policy = st.slider(
                "Import Policy & Tariff Impact (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.2,
                step=0.1,
                help="Impact of import policies on domestic prices"
            )
        
        # Calculate total policy impact
        total_impact = monetary_policy + subsidy_impact + supply_chain + import_policy
        
        # Create policy parameters dictionary
        policy_params = {
            'CPI_Value': total_impact  # Overall impact on CPI
        }
        
        # Run simulation with policy impacts
        if st.button("Run Simulation"):
            with st.spinner("Simulating policy impacts..."):
                # Simulate policy impact
                adjusted_df = simulate_policy_impact(prediction_df, policy_params)
                
                # Visualize base vs adjusted predictions
                fig = make_subplots(specs=[[{"secondary_y": False}]])
                
                # Add historical data
                historical = adjusted_df[adjusted_df['Predicted'] == False]
                fig.add_trace(
                    go.Scatter(
                        x=historical['Date'],
                        y=historical['YoY_Change'],
                        name="Historical",
                        line=dict(color="blue")
                    )
                )
                
                # Add base prediction
                base_prediction = prediction_df[prediction_df['Predicted'] == True]
                fig.add_trace(
                    go.Scatter(
                        x=base_prediction['Date'],
                        y=base_prediction['YoY_Change'],
                        name="Base Prediction",
                        line=dict(color="red", dash="dash")
                    )
                )
                
                # Add adjusted prediction
                adjusted_prediction = adjusted_df[adjusted_df['Predicted'] == True]
                fig.add_trace(
                    go.Scatter(
                        x=adjusted_prediction['Date'],
                        y=adjusted_prediction['YoY_Change'],
                        name="With Policy Intervention",
                        line=dict(color="green")
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title="Inflation Forecast with Policy Interventions",
                    xaxis_title="Date",
                    yaxis_title="YoY Inflation (%)",
                    legend_title="Scenario",
                    height=500,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display impact metrics
                final_base = base_prediction['YoY_Change'].iloc[-1]
                final_adjusted = adjusted_prediction['YoY_Change'].iloc[-1]
                reduction = final_base - final_adjusted
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Projected Inflation (No Intervention)",
                        f"{final_base:.2f}%"
                    )
                
                with metric_col2:
                    st.metric(
                        "Projected Inflation (With Intervention)",
                        f"{final_adjusted:.2f}%"
                    )
                
                with metric_col3:
                    st.metric(
                        "Inflation Reduction",
                        f"{reduction:.2f} pp",
                        delta=f"-{(reduction / final_base * 100):.1f}%"
                    )
                
                # Policy recommendation
                st.subheader("Policy Effectiveness Analysis")
                
                if reduction <= 0.5:
                    recommendation = "The selected policy interventions show minimal impact. Consider stronger measures or different policy combinations for more significant inflation control."
                elif reduction <= 1.0:
                    recommendation = "The selected policy mix shows moderate effectiveness. There is room for optimization to achieve better inflation control."
                else:
                    recommendation = "The selected policy interventions show strong effectiveness in controlling inflation. This policy mix could be considered for implementation."
                
                st.write(recommendation)
                
                # Display policy contribution breakdown
                st.subheader("Policy Contribution Breakdown")
                
                # Calculate relative contributions
                policy_contributions = {
                    "Monetary Policy": monetary_policy / total_impact * 100,
                    "Subsidies & Price Controls": subsidy_impact / total_impact * 100,
                    "Supply Chain Improvements": supply_chain / total_impact * 100,
                    "Import & Tariff Policies": import_policy / total_impact * 100
                }
                
                # Create contribution chart
                contribution_df = pd.DataFrame({
                    "Policy": list(policy_contributions.keys()),
                    "Contribution (%)": list(policy_contributions.values())
                })
                
                fig = px.pie(
                    contribution_df,
                    values="Contribution (%)",
                    names="Policy",
                    title="Relative Contribution of Policy Measures",
                    hole=0.4
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Comprehensive Analysis
    with tab5:
        st.header("Comprehensive Inflation Analysis Dashboard")
        
        # Create a comprehensive dashboard with key metrics
        st.markdown("""
        This dashboard provides a holistic view of current inflation dynamics,
        combining key metrics and visualizations for quick assessment and decision-making.
        """)
        
        # Current inflation metrics
        st.subheader("Current Inflation Metrics (Combined)")
        
        # Get latest data
        latest_date = cpi_df['Date'].max()
        latest_data = cpi_df[(cpi_df['Date'] == latest_date) & (cpi_df['Region'] == 'Combined')]
        
        # Display key metrics in columns
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            general_inflation = latest_data[latest_data['Category'] == 'General Index']['YoY_Change'].values[0]
            st.metric(
                "Headline Inflation",
                f"{general_inflation:.2f}%",
                delta=f"{general_inflation - latest_data[latest_data['Category'] == 'General Index']['MoM_Change'].values[0]:.2f} pp YoY"
            )
        
        with metrics_col2:
            food_inflation = latest_data[latest_data['Category'] == 'Food and beverages']['YoY_Change'].values[0]
            st.metric(
                "Food Inflation",
                f"{food_inflation:.2f}%",
                delta=f"{food_inflation - general_inflation:.2f} pp vs Headline"
            )
        
        with metrics_col3:
            fuel_inflation = latest_data[latest_data['Category'] == 'Fuel and light']['YoY_Change'].values[0]
            st.metric(
                "Fuel Inflation",
                f"{fuel_inflation:.2f}%",
                delta=f"{fuel_inflation - general_inflation:.2f} pp vs Headline"
            )
        
        with metrics_col4:
            core_categories = ['Clothing and footwear', 'Housing', 'Miscellaneous']
            core_inflation = latest_data[latest_data['Category'].isin(core_categories)]['YoY_Change'].mean()
            st.metric(
                "Core Inflation",
                f"{core_inflation:.2f}%",
                delta=f"{core_inflation - general_inflation:.2f} pp vs Headline"
            )
        
        # Create a comprehensive multi-chart dashboard
        st.subheader("Inflation Overview")
        
        # Create composite visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Category-wise Inflation (Latest)",
                "Regional Comparison",
                "Monthly Trend (General Index)",
                "Income Group Impact"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Category-wise latest inflation (top left)
        category_data = latest_data.sort_values('YoY_Change', ascending=False)
        fig.add_trace(
            go.Bar(
                x=category_data['Category'],
                y=category_data['YoY_Change'],
                marker_color='blue',
                name="Category Inflation"
            ),
            row=1, col=1
        )
        
        # 2. Regional comparison (top right)
        region_data = cpi_df[
            (cpi_df['Date'] == latest_date) & 
            (cpi_df['Category'] == 'General Index')
        ]
        fig.add_trace(
            go.Bar(
                x=region_data['Region'],
                y=region_data['YoY_Change'],
                marker_color='green',
                name="Regional Inflation"
            ),
            row=1, col=2
        )
        
        # 3. Monthly trend for last 12 months (bottom left)
        trend_data = cpi_df[
            (cpi_df['Category'] == 'General Index') & 
            (cpi_df['Region'] == 'Combined')
        ].tail(12)
        fig.add_trace(
            go.Scatter(
                x=trend_data['Date'],
                y=trend_data['YoY_Change'],
                mode='lines+markers',
                name="12-Month Trend",
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # 4. Income group impact (bottom right)
        fig.add_trace(
            go.Bar(
                x=income_impact_df['Income_Group'],
                y=income_impact_df['Total_Impact'],
                marker_color='purple',
                name="Income Impact"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Comprehensive Inflation Dashboard"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Category", row=1, col=1)
        fig.update_yaxes(title_text="YoY Inflation (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Region", row=1, col=2)
        fig.update_yaxes(title_text="YoY Inflation (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="YoY Inflation (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Income Group", row=2, col=2)
        fig.update_yaxes(title_text="Effective Inflation (%)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Policy recommendation section
        st.subheader("Policy Implications & Recommendations")
        
        # Create columns for key focus areas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Key Areas of Concern")
            
            # Identify top inflation categories
            top_categories = category_data.head(2)['Category'].tolist()
            
            # Identify most affected regions
            top_regions = region_data.sort_values('YoY_Change', ascending=False).head(2)['Region'].tolist()
            
            # Create bullet points for areas of concern
            concerns = [
                f"High inflation in {', '.join(top_categories)} categories.",
                f"Elevated inflation levels in {', '.join(top_regions)} regions.",
                f"Disproportionate impact on {income_impact_df.iloc[income_impact_df['Total_Impact'].idxmax()]['Income_Group']} income group."
            ]
            
            for concern in concerns:
                st.markdown(f"- {concern}")
        
        with col2:
            st.markdown("### Policy Recommendations")
            
            # Generate recommendations based on data
            if food_inflation > general_inflation + 1:
                food_rec = "Implement targeted food price stabilization measures through buffer stock management and supply chain improvements."
            else:
                food_rec = "Monitor food prices and maintain current supply chain efficiency."
            
            if fuel_inflation > general_inflation + 1:
                fuel_rec = "Consider adjusting fuel taxes or subsidies to moderate the impact on overall inflation."
            else:
                fuel_rec = "Maintain current fuel pricing policies while monitoring global energy markets."
            
            if core_inflation > general_inflation:
                core_rec = "Address structural factors driving core inflation through monetary policy measures."
            else:
                core_rec = "Focus on addressing food and fuel price volatility while maintaining current monetary stance."
            
            recommendations = [food_rec, fuel_rec, core_rec]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Download section
        st.subheader("Data Download")
        
        # Prepare data for download
        download_tab1, download_tab2 = st.tabs(["CPI Data", "Analysis Results"])
        
        with download_tab1:
            st.dataframe(cpi_df.head(100))
            
            # Convert to CSV
            csv = cpi_df.to_csv(index=False)
            st.download_button(
                label="Download Full CPI Data",
                data=csv,
                file_name="cpi_data.csv",
                mime="text/csv"
            )
        
        with download_tab2:
            # Prepare summary tables
            latest_summary = latest_data[['Category', 'YoY_Change', 'MoM_Change']]
            latest_summary.columns = ['Category', 'YoY Inflation (%)', 'MoM Inflation (%)']
            
            st.write("Latest Inflation Summary")
            st.dataframe(latest_summary)
            
            st.write("Income Group Impact Analysis")
            st.dataframe(income_impact_df)
            
            # Combine summary tables for download
            analysis_results = {
                "Latest Inflation": latest_summary.to_dict('records'),
                "Income Impact": income_impact_df.to_dict('records'),
                "Regional Data": state_df.to_dict('records')
            }
            
            # Convert to JSON
            json_results = json.dumps(analysis_results, indent=2)
            st.download_button(
                label="Download Analysis Results",
                data=json_results,
                file_name="inflation_analysis_results.json",
                mime="application/json"
            )

# Run the application
if __name__ == "__main__":
    build_app()
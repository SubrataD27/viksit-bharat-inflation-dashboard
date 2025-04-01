# Inflation Insights Dashboard

## Understanding Price Dynamics for Viksit Bharat

**Author:** Subrata Dhibar

## 📊 Project Overview

The **Inflation Insights Dashboard** is a comprehensive interactive web application that provides data-driven inflation analysis for policymakers working toward a developed India ("Viksit Bharat"). This tool visualizes and analyzes Consumer Price Index (CPI) trends across regions, sectors, and demographic groups, enabling targeted interventions and informed policy formulation.

This dashboard was developed as part of the **[Hackathon Name]** organized by **[Organization Name]**.

## ✨ Key Features

- **Comprehensive Inflation Trends Analysis:** Track CPI values and year-over-year changes with interactive time series visualizations
- **Regional Comparison:** Explore state-wise inflation variations with interactive choropleth maps
- **Income Group Impact Analysis:** Understand how inflation affects different socioeconomic segments
- **Predictive Modeling:** View 12-month inflation forecasts based on historical patterns
- **Policy Simulation:** Model the impact of various policy interventions on future inflation rates
- **Seasonal Pattern Detection:** Identify monthly and seasonal inflation trends
- **Urban-Rural Gap Analysis:** Compare inflation dynamics between urban and rural areas

## 🛠️ Technology Stack

- **Python 3.x:** Core programming language
- **Streamlit:** Web application framework
- **Pandas & NumPy:** Data manipulation and numerical computing
- **Plotly & Matplotlib:** Interactive and static data visualization
- **scikit-learn:** Machine learning for predictive modeling
- **Seaborn:** Enhanced statistical data visualization

## 📋 Installation and Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/inflation-insights-dashboard.git
   cd inflation-insights-dashboard
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate
   
   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard:**
   Open a web browser and navigate to `http://localhost:8501`



## 🔍 Features in Detail

### 1. Inflation Trends Analysis

- **Time Series Visualization:** Track CPI values and YoY changes with interactive line charts
- **Heatmap Analysis:** Identify monthly patterns and seasonal trends
- **Category Comparison:** Compare inflation across different expenditure categories

### 2. Regional Analysis

- **Choropleth Map:** Visualize state-wise inflation variations
- **Urban-Rural Comparison:** Compare inflation rates between urban and rural areas
- **Regional Gap Analysis:** Track the urban-rural inflation differential over time

### 3. Impact on Income Groups

- **Differential Impact Analysis:** Understand how inflation affects different income segments
- **Category-wise Impact:** Break down inflation impact by expenditure category for each income group
- **Food Inflation Focus:** Special focus on food inflation impact on low-income groups

### 4. Predictions & Simulations

- **12-Month Forecast:** View inflation projections based on historical trends
- **Policy Simulation:** Model how different policy interventions might affect inflation
- **Impact Metrics:** Quantify potential inflation reduction through policy measures
- **Contribution Analysis:** Understand the relative contribution of different policy levers

### 5. Comprehensive Dashboard

- **Key Metrics:** View current headline inflation, food inflation, and fuel inflation
- **Trend Indicators:** Track month-over-month and year-over-year changes
- **Visual Summary:** Get a holistic view of the inflation landscape

## 💾 Data Sources

The dashboard currently uses synthetic data that mimics the structure and patterns of actual CPI data. In a production environment, the system would connect to:

- **Consumer Price Index (CPI) data** from the Ministry of Statistics and Programme Implementation (MOSPI)
- **eSankhyiki Portal** for official government statistics
- **State-wise economic indicators** from various government sources

## 🚀 Usage Guide

### Navigation

The dashboard is organized into five main tabs:

1. **Inflation Trends:** Explore historical CPI values and inflation rates
2. **Regional Analysis:** Analyze state-wise and urban-rural inflation patterns
3. **Impact on Income Groups:** Understand how inflation affects different socioeconomic segments
4. **Predictions & Simulations:** View forecasts and simulate policy impacts
5. **Comprehensive Analysis:** Get a holistic view of current inflation metrics

### Filter Controls

Each section includes relevant filters to customize the analysis:

- **Region Selection:** Choose between Rural, Urban, and Combined data
- **Category Selection:** Focus on specific expenditure categories
- **Trend Type Selection:** Toggle between CPI values and year-over-year changes

### Policy Simulation

To simulate policy impacts:

1. Navigate to the "Predictions & Simulations" tab
2. Adjust the sliders for different policy interventions
3. Click "Run Simulation" to view the projected impact
4. Review the effectiveness metrics and policy contribution breakdown

## 🔧 Customization and Extension

### Adding Real Data

To replace synthetic data with real CPI data:

1. Prepare your CSV files with the required structure (see data loader function)
2. Modify the `load_cpi_data()` function to load from your CSV files
3. Update any data-specific logic if your structure differs from the synthetic data

### Adding New Visualizations

To add new visualization types:

1. Create the data processing function in the appropriate section
2. Add the visualization code using Plotly or other libraries
3. Update the UI to include your new visualization

### Extending Policy Simulations

To enhance the policy simulation capabilities:

1. Add new policy parameters to the simulation controls
2. Update the `simulate_policy_impact()` function to incorporate your new parameters
3. Modify the visualization code to show the impact of your additional policies

## 🤝 Contributing

Contributions to improve the Inflation Insights Dashboard are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Ministry of Statistics and Programme Implementation](https://mospi.gov.in/) for the CPI data structure
- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [Plotly](https://plotly.com/) for the interactive visualization capabilities
- Innovate with GoIStats Hackathon for the opportunity to build this solution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
import json
import os
from folium.plugins import HeatMap
from sklearn.preprocessing import MinMaxScaler

class VisualizationEngine:
    def __init__(self, data_path="data/processed/development_index.csv"):
        self.data = pd.read_csv(data_path)
        self.india_states_geo = json.load(open("assets/india_states.geojson", "r"))
        
        # Create assets directory if it doesn't exist
        if not os.path.exists("assets"):
            os.makedirs("assets")
            
    def create_choropleth_map(self):
        """Create a choropleth map of development index"""
        # Center coordinates for India
        india_center = [20.5937, 78.9629]
        
        # Create map
        m = folium.Map(location=india_center, zoom_start=5, tiles="CartoDB positron")
        
        # Add choropleth layer
        choropleth = folium.Choropleth(
            geo_data=self.india_states_geo,
            name="Development Index",
            data=self.data,
            columns=["State", "Development_Index"],
            key_on="feature.properties.NAME_1",
            fill_color="YlGnBu",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Development Index",
            highlight=True
        ).add_to(m)
        
        # Add tooltips
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(
                fields=["NAME_1"],
                aliases=["State"],
                style="""
                    background-color: #F0EFEF;
                    border: 1px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """
            )
        )
        
        # Add hover information
        style_function = lambda x: {
            "fillColor": "#ffffff",
            "color": "#000000",
            "fillOpacity": 0.1,
            "weight": 0.1
        }
        highlight_function = lambda x: {
            "fillColor": "#000000",
            "color": "#000000",
            "fillOpacity": 0.50,
            "weight": 0.1
        }
        
        # Join data with geojson
        state_data = {}
        for _, row in self.data.iterrows():
            state_data[row["State"]] = {
                "Development_Index": round(row["Development_Index"], 3) if not np.isnan(row["Development_Index"]) else "No data",
                "LFPR": round(row["LFPR"], 1) if "LFPR" in self.data.columns and not np.isnan(row["LFPR"]) else "No data",
                "MPCE": round(row["MPCE"], 0) if "MPCE" in self.data.columns and not np.isnan(row["MPCE"]) else "No data",
                "Per_Capita_GSDP": round(row["Per_Capita_GSDP"], 0) if "Per_Capita_GSDP" in self.data.columns and not np.isnan(row["Per_Capita_GSDP"]) else "No data"
            }
        
        # Add state-specific data
        folium.GeoJson(
            self.india_states_geo,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_1"],
                aliases=["State"],
                style="""
                    background-color: #F0EFEF;
                    border: 1px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """
            ),
            popup=folium.GeoJsonPopup(
                fields=["NAME_1"],
                aliases=["State"],
                localize=True,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 1px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """
            )
        ).add_to(m)
        
        # Save map
        m.save("assets/development_map.html")
        
        return m
    
    def create_gap_chart(self):
        """Create a development gap visualization"""
        # Sort by development gap
        sorted_data = self.data.sort_values("Development_Gap", ascending=False)
        
        # Filter out NaN values
        filtered_data = sorted_data[~sorted_data["Development_Gap"].isna()]
        
        # Take top 15 states with largest gaps
        top_gap_states = filtered_data.head(15)
        
        # Create bar chart
        fig = px.bar(
            top_gap_states,
            x="State",
            y="Development_Gap",
            color="Development_Gap",
            color_continuous_scale="Reds",
            title="States with Largest Development Gaps",
            labels={"Development_Gap": "Gap from Target", "State": ""},
            height=500
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        # Save figure
        fig.write_html("assets/gap_chart.html")
        
        return fig
    
    def create_correlation_heatmap(self):
        """Create a correlation heatmap"""
        # Load correlation data
        corr_data = pd.read_csv("data/processed/correlations.csv", index_col=0)
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_data,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation between Development Indicators",
            labels=dict(x="Indicator", y="Indicator", color="Correlation"),
            height=600,
            width=700
        )
        
        # Customize layout
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white"
        )
        
        # Save figure
        fig.write_html("assets/correlation_heatmap.html")
        
        return fig
    
    def create_projection_chart(self):
        """Create a development projection chart"""
        # Load projection data
        projection_data = pd.read_csv("data/processed/projections.csv")
        
        # Select top 10 states by current development index
        top_states = self.data.sort_values("Development_Index", ascending=False).head(10)["State"].tolist()
        
        # Filter projection data for these states
        top_projections = projection_data[projection_data["State"].isin(top_states)]
        
        # Create line chart
        fig = px.line(
            top_projections,
            x="Year",
            y="Projected_Index",
            color="State",
            title="Development Index Projections (2025-2030)",
            labels={"Projected_Index": "Development Index", "Year": "Year"},
            height=500
        )
        
        # Add target line
        fig.add_shape(
            type="line",
            x0=2025,
            x1=2030,
            y0=0.8,
            y1=0.8,
            line=dict(color="red", width=2, dash="dash"),
            name="Viksit Bharat Target"
        )
        
        # Add annotation for target line
        fig.add_annotation(
            x=2028,
            y=0.82,
            text="Viksit Bharat Target",
            showarrow=False,
            font=dict(color="red")
        )
        
        # Customize layout
        fig.update_layout(
            xaxis=dict(tickmode="linear", dtick=1),
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        # Save figure
        fig.write_html("assets/projection_chart.html")
        
        return fig

    def create_sector_comparative_chart(self):
        """Create a sector comparative analysis chart"""
        # Prepare data
        sector_data = self.data[["State", "LFPR", "MPCE", "Per_Capita_GSDP"]]
        sector_data = sector_data.dropna()
        
        # Normalize indicators for radar chart
        scaler = MinMaxScaler()
        sector_data[["LFPR_norm", "MPCE_norm", "GSDP_norm"]] = scaler.fit_transform(
            sector_data[["LFPR", "MPCE", "Per_Capita_GSDP"]]
        )
        
        # Select top 5 states by development index
        top_states = self.data.sort_values("Development_Index", ascending=False).head(5)["State"].tolist()
        
        # Filter data for these states
        top_sector_data = sector_data[sector_data["State"].isin(top_states)]
        
        # Create radar chart
        fig = go.Figure()
        
        for state in top_states:
            state_data = top_sector_data[top_sector_data["State"] == state]
            if not state_data.empty:
                fig.add_trace(go.Scatterpolar(
                    r=[
                        state_data["LFPR_norm"].values[0],
                        state_data["MPCE_norm"].values[0],
                        state_data["GSDP_norm"].values[0],
                        state_data["LFPR_norm"].values[0]  # Close the loop
                    ],
                    theta=["Labor Force Participation", "Consumption", "Economic Output", "Labor Force Participation"],
                    fill="toself",
                    name=state
                ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Sector Balance Analysis for Top States",
            height=500,
            width=700
        )
        
        # Save figure
        fig.write_html("assets/sector_comparative.html")
        
        return fig
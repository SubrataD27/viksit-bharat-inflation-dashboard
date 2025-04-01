# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import dash_bootstrap_components as dbc
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# import os
# import json
# from src.data_processing import DataProcessor
# from src.analytics import AnalyticsEngine
# from src.visualization import VisualizationEngine

# # Initialize data processing pipeline
# print("Initializing data processing...")
# processor = DataProcessor()
# integrated_data = processor.integrate_datasets()

# # Run analytics
# print("Running analytics...")
# analytics_engine = AnalyticsEngine()
# analyzed_data = analytics_engine.create_development_index()
# correlations = analytics_engine.analyze_correlations()
# projections = analytics_engine.project_development_trends()
# gap_analysis = analytics_engine.identify_key_gaps()

# # Create visualizations
# print("Creating visualizations...")
# viz_engine = VisualizationEngine()
# viz_engine.create_choropleth_map()
# viz_engine.create_gap_chart()
# viz_engine.create_correlation_heatmap()
# viz_engine.create_projection_chart()
# viz_engine.create_sector_comparative_chart()

# # Load processed data for dashboard
# data = pd.read_csv("data/processed/gap_analysis.csv")
# projections = pd.read_csv("data/processed/projections.csv")

# # Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# server = app.server

# app.title = "Viksit Bharat Insights Hub"

# # Load map HTML
# with open("assets/development_map.html", "r") as f:
#     map_html = f.read()

# # Define layout
# app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             html.H1("Viksit Bharat Insights Hub", className="text-center p-3 mb-2 bg-primary text-white")
#         ])
#     ]),
    
#     dbc.Row([
#         dbc.Col([
#             html.H4("Bridging Development Gaps Through Integrated Data Analytics", className="text-center text-muted mb-4")
#         ])
#     ]),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Index Map"),
#                 dbc.CardBody([
#                     html.Iframe(
#                         id="choropleth-map",
#                         srcDoc=map_html,
#                         style={"width": "100%", "height": "450px", "border": "none"}
#                     )
#                 ])
#             ])
#         ], width=12)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("State Selection"),
#                 dbc.CardBody([
#                     dcc.Dropdown(
#                         id="state-dropdown",
#                         options=[{"label": state, "value": state} for state in data["State"].unique()],
#                         value="All States",
#                         clearable=False
#                     )
#                 ])
#             ])
#         ], width=3),
        
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Gap Analysis"),
#                 dbc.CardBody([
#                     dcc.Graph(id="gap-chart")
#                 ])
#             ])
#         ], width=9)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("State Development Profile"),
#                 dbc.CardBody([
#                     html.Div(id="state-profile")
#                 ])
#             ])
#         ], width=4),
        
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Projection"),
#                 dbc.CardBody([
#                     dcc.Graph(id="projection-chart")
#                 ])
#             ])
#         ], width=8)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Sector Balance Analysis"),
#                 dbc.CardBody([
#                     html.Iframe(
#                         id="sector-comparative",
#                         srcDoc=open("assets/sector_comparative.html", "r").read(),
#                         style={"width": "100%", "height": "450px", "border": "none"}
#                     )
#                 ])
#             ])
#         ], width=6),
        
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Indicator Correlations"),
#                 dbc.CardBody([
#                     html.Iframe(
#                         id="correlation-heatmap",
#                         srcDoc=open("assets/correlation_heatmap.html", "r").read(),
#                         style={"width": "100%", "height": "450px", "border": "none"}
#                     )
#                 ])
#             ])
#         ], width=6)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             html.Div([
#                 html.H5("About This Dashboard", className="text-center"),
#                 html.P([
#                     "The Viksit Bharat Insights Hub provides an integrated view of development indicators across India, ",
                    






# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import dash_bootstrap_components as dbc
# import plotly.express as px
# import plotly.graph_objects as go
# import pandas as pd
# import os
# import json
# from src.data_processing import DataProcessor
# from src.analytics import AnalyticsEngine
# from src.visualization import VisualizationEngine

# # Initialize data processing pipeline
# print("Initializing data processing...")
# processor = DataProcessor()
# integrated_data = processor.integrate_datasets()

# # Run analytics
# print("Running analytics...")
# analytics_engine = AnalyticsEngine()
# analyzed_data = analytics_engine.create_development_index()
# correlations = analytics_engine.analyze_correlations()
# projections = analytics_engine.project_development_trends()
# gap_analysis = analytics_engine.identify_key_gaps()

# # Create visualizations
# print("Creating visualizations...")
# viz_engine = VisualizationEngine()
# viz_engine.create_choropleth_map()
# viz_engine.create_gap_chart()
# viz_engine.create_correlation_heatmap()
# viz_engine.create_projection_chart()
# viz_engine.create_sector_comparative_chart()

# # Load processed data for dashboard
# data = pd.read_csv("data/processed/gap_analysis.csv")
# projections = pd.read_csv("data/processed/projections.csv")

# # Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# server = app.server

# app.title = "Viksit Bharat Insights Hub"

# # Load map HTML
# with open("assets/development_map.html", "r") as f:
#     map_html = f.read()

# # Define layout
# app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             html.H1("Viksit Bharat Insights Hub", className="text-center p-3 mb-2 bg-primary text-white")
#         ])
#     ]),
    
#     dbc.Row([
#         dbc.Col([
#             html.H4("Bridging Development Gaps Through Integrated Data Analytics", className="text-center text-muted mb-4")
#         ])
#     ]),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Index Map"),
#                 dbc.CardBody([
#                     html.Iframe(
#                         id="choropleth-map",
#                         srcDoc=map_html,
#                         style={"width": "100%", "height": "450px", "border": "none"}
#                     )
#                 ])
#             ])
#         ], width=12)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("State Selection"),
#                 dbc.CardBody([
#                     dcc.Dropdown(
#                         id="state-dropdown",
#                         options=[{"label": state, "value": state} for state in data["State"].unique()],
#                         value="All States",
#                         clearable=False
#                     )
#                 ])
#             ])
#         ], width=3),
        
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Gap Analysis"),
#                 dbc.CardBody([
#                     dcc.Graph(id="gap-chart")
#                 ])
#             ])
#         ], width=9)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("State Development Profile"),
#                 dbc.CardBody([
#                     html.Div(id="state-profile")
#                 ])
#             ])
#         ], width=4),
        
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Projection"),
#                 dbc.CardBody([
#                     dcc.Graph(id="projection-chart")
#                 ])
#             ])
#         ], width=8)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Sector Balance Analysis"),
#                 dbc.CardBody([
#                     html.Iframe(
#                         id="sector-comparative",
#                         srcDoc=open("assets/sector_comparative.html", "r").read(),
#                         style={"width": "100%", "height": "450px", "border": "none"}
#                     )
#                 ])
#             ])
#         ], width=6),
        
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("Development Indicator Correlations"),
#                 dbc.CardBody([
#                     html.Iframe(
#                         id="correlation-heatmap",
#                         srcDoc=open("assets/correlation_heatmap.html", "r").read(),
#                         style={"width": "100%", "height": "450px", "border": "none"}
#                     )
#                 ])
#             ])
#         ], width=6)
#     ], className="mb-4"),
    
#     dbc.Row([
#         dbc.Col([
#             html.Div([
#                 html.H5("About This Dashboard", className="text-center"),
#                 html.P([
#                     "The Viksit Bharat Insights Hub provides an integrated view of development indicators across India, "
#                     "enabling policymakers, researchers, and citizens to analyze gaps, correlations, and projected trends "
#                     "to drive informed decision-making."
#                 ])
#             ])
#         ])
#     ])
# ])

# # Define Callbacks
# @app.callback(
#     Output("gap-chart", "figure"),
#     [Input("state-dropdown", "value")]
# )
# def update_gap_chart(selected_state):
#     filtered_data = data if selected_state == "All States" else data[data["State"] == selected_state]
#     fig = px.bar(filtered_data, x="Sector", y="Gap", color="Sector", title=f"Development Gaps in {selected_state}")
#     return fig

# @app.callback(
#     Output("projection-chart", "figure"),
#     [Input("state-dropdown", "value")]
# )
# def update_projection_chart(selected_state):
#     filtered_projections = projections if selected_state == "All States" else projections[projections["State"] == selected_state]
#     fig = px.line(filtered_projections, x="Year", y="Projected Index", color="Sector", title=f"Development Projections for {selected_state}")
#     return fig

# # Run the app
# if __name__ == "__main__":
#     app.run_server(debug=True)



import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import json
from src.data_processing import DataProcessor
from src.analytics import AnalyticsEngine
from src.visualization import VisualizationEngine

# Initialize data processing pipeline
print("Initializing data processing...")
processor = DataProcessor()
integrated_data = processor.integrate_datasets()

# Run analytics
print("Running analytics...")
analytics_engine = AnalyticsEngine()
analyzed_data = analytics_engine.create_development_index()
correlations = analytics_engine.analyze_correlations()
projections = analytics_engine.project_development_trends()
gap_analysis = analytics_engine.identify_key_gaps()

# Create visualizations
print("Creating visualizations...")
viz_engine = VisualizationEngine()
viz_engine.create_choropleth_map()
viz_engine.create_gap_chart()
viz_engine.create_correlation_heatmap()
viz_engine.create_projection_chart()
viz_engine.create_sector_comparative_chart()

# Load processed data for dashboard
data = pd.read_csv("data/processed/gap_analysis.csv")
projections = pd.read_csv("data/processed/projections.csv")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.title = "Viksit Bharat Insights Hub"

# Load map HTML
with open("assets/development_map.html", "r") as f:
    map_html = f.read()

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Viksit Bharat Insights Hub", className="text-center p-3 mb-2 bg-primary text-white")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Bridging Development Gaps Through Integrated Data Analytics", className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Development Index Map"),
                dbc.CardBody([
                    html.Iframe(
                        id="choropleth-map",
                        srcDoc=map_html,
                        style={"width": "100%", "height": "450px", "border": "none"}
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("State Selection"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="state-dropdown",
                        options=[{"label": state, "value": state} for state in data["State"].unique()],
                        value="All States",
                        clearable=False
                    )
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Development Gap Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="gap-chart")
                ])
            ])
        ], width=9)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("State Development Profile"),
                dbc.CardBody([
                    html.Div(id="state-profile")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Development Projection"),
                dbc.CardBody([
                    dcc.Graph(id="projection-chart")
                ])
            ])
        ], width=8)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sector Balance Analysis"),
                dbc.CardBody([
                    html.Iframe(
                        id="sector-comparative",
                        srcDoc=open("assets/sector_comparative.html", "r").read(),
                        style={"width": "100%", "height": "450px", "border": "none"}
                    )
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Development Indicator Correlations"),
                dbc.CardBody([
                    html.Iframe(
                        id="correlation-heatmap",
                        srcDoc=open("assets/correlation_heatmap.html", "r").read(),
                        style={"width": "100%", "height": "450px", "border": "none"}
                    )
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("About This Dashboard", className="text-center"),
                html.P([
                    "The Viksit Bharat Insights Hub provides an integrated view of development indicators across India, ",
                    "combining labor, consumption, and economic growth data to identify development gaps and opportunities. ",
                    "This platform aims to support evidence-based policymaking to accelerate India's journey toward becoming ",
                    "a developed nation by 2047."
                ]),
                html.P([
                    "Data Sources: Annual Periodic Labour Force Survey (PLFS), ",
                    "Household Consumer Expenditure Survey (HCES), and National Accounts Statistics (GDP)."
                ], className="text-muted")
            ], className="text-center p-3")
        ], width=12)
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output("gap-chart", "figure"),
    [Input("state-dropdown", "value")]
)
def update_gap_chart(selected_state):
    if selected_state == "All States":
        # Show top 10 states with largest gaps
        filtered_df = data.sort_values("Development_Gap", ascending=False).head(10)
        title = "Top 10 States with Largest Development Gaps"
    else:
        # Show selected state and its neighbors for comparison
        filtered_df = data[data["State"] == selected_state]
        title = f"Development Gap Analysis: {selected_state}"
    
    fig = px.bar(
        filtered_df,
        x="State",
        y="Development_Gap",
        color="Development_Gap",
        color_continuous_scale="Reds",
        title=title,
        labels={"Development_Gap": "Gap from Target", "State": ""}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

@app.callback(
    Output("projection-chart", "figure"),
    [Input("state-dropdown", "value")]
)
def update_projection_chart(selected_state):
    if selected_state == "All States":
        # Show projections for top 5 states
        top_states = data.sort_values("Development_Index", ascending=False).head(5)["State"].tolist()
        filtered_df = projections[projections["State"].isin(top_states)]
        title = "Development Index Projections for Top 5 States (2025-2030)"
    else:
        # Show projection for selected state
        filtered_df = projections[projections["State"] == selected_state]
        title = f"Development Index Projection: {selected_state} (2025-2030)"
    
    fig = px.line(
        filtered_df,
        x="Year",
        y="Projected_Index",
        color="State",
        title=title,
        labels={"Projected_Index": "Development Index", "Year": "Year"}
    )
    
    # Add target line
    fig.add_shape(
        type="line",
        x0=2025,
        x1=2030,
        y0=0.8,
        y1=0.8,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add annotation
    fig.add_annotation(
        x=2028,
        y=0.82,
        text="Viksit Bharat Target",
        showarrow=False,
        font=dict(color="red")
    )
    
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

@app.callback(
    Output("state-profile", "children"),
    [Input("state-dropdown", "value")]
)
def update_state_profile(selected_state):
    if selected_state == "All States":
        return html.Div([
            html.H4("National Overview", className="text-center mb-3"),
            html.P("Select a specific state to view its development profile.")
        ])
    else:
        # Get state data
        state_data = data[data["State"] == selected_state].iloc[0]
        
        # Create profile
        return html.Div([
            html.H4(f"{selected_state}", className="text-center mb-3"),
            
            html.Div([
                html.H6("Development Index", className="font-weight-bold"),
                html.Div([
                    html.Span(f"{state_data['Development_Index']:.3f}", 
                             className="h3 text-primary"),
                    html.Span(f" / 1.00", className="text-muted")
                ], className="mb-2")
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Labor Indicators", className="font-weight-bold"),
                        html.P([
                            html.Span("LFPR: ", className="font-weight-bold"),
                            f"{state_data['LFPR']:.1f}%"
                        ]),
                        html.P([
                            html.Span("WPR: ", className="font-weight-bold"),
                            f"{state_data['WPR']:.1f}%"
                        ]),
                        html.P([
                            html.Span("Unemployment: ", className="font-weight-bold"),
                            f"{state_data['UR']:.1f}%"
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H6("Economic Indicators", className="font-weight-bold"),
                        html.P([
                            html.Span("MPCE: ", className="font-weight-bold"),
                            f"₹{state_data['MPCE']:.0f}"
                        ]),
                        html.P([
                            html.Span("GSDP Growth: ", className="font-weight-bold"),
                            f"{state_data['Growth_Rate']:.1f}%"
                        ]),
                        html.P([
                            html.Span("Per Capita GSDP: ", className="font-weight-bold"),
                            f"₹{state_data['Per_Capita_GSDP']:.0f}"
                        ])
                    ])
                ], width=6)
            ]),
            
            html.Div([
                html.H6("Development Gap Assessment", className="font-weight-bold mt-3"),
                dbc.Progress(
                    value=state_data["Development_Gap"] * 100, 
                    color="danger", 
                    className="mb-2",
                    style={"height": "20px"}
                ),
                html.P(f"Gap from target: {state_data['Development_Gap']:.3f}")
            ])
        ])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
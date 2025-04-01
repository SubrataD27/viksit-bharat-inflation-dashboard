import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

class DashboardBuilder:
    def __init__(self, data_path="data/processed/gap_analysis.csv"):
        self.data = pd.read_csv(data_path)
        self.projections = pd.read_csv("data/processed/projections.csv")
        
        # Initialize the Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Viksit Bharat Insights Hub"
    
    def create_layout(self):
        """Create the dashboard layout"""
        # Load map HTML
        with open("assets/development_map.html", "r") as f:
            map_html = f.read()
            
        # Create layout
        self.app.layout = dbc.Container([
            # Header
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
            
            # Map
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
            
            # State selection and gap analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("State Selection"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="state-dropdown",
                                options=[{"label": state, "value": state} for state in self.data["State"].unique()],
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
            
            # State profile and projection
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
            
            # Sector analysis and correlations
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
            
            # Footer
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
        
        return self.app
    
    def register_callbacks(self):
        """Register all dashboard callbacks"""
        
        @self.app.callback(
            Output("gap-chart", "figure"),
            [Input("state-dropdown", "value")]
        )
        def update_gap_chart(selected_state):
            if selected_state == "All States":
                # Show top 10 states with largest gaps
                filtered_df = self.data.sort_values("Development_Gap", ascending=False).head(10)
                title = "Top 10 States with Largest Development Gaps"
            else:
                # Show selected state and its neighbors for comparison
                filtered_df = self.data[self.data["State"] == selected_state]
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
        
        @self.app.callback(
            Output("projection-chart", "figure"),
            [Input("state-dropdown", "value")]
        )
        def update_projection_chart(selected_state):
            if selected_state == "All States":
                # Show projections for top 5 states
                top_states = self.data.sort_values("Development_Index", ascending=False).head(5)["State"].tolist()
                filtered_df = self.projections[self.projections["State"].isin(top_states)]
                title = "Development Index Projections for Top 5 States (2025-2030)"
            else:
                # Show projection for selected state
                filtered_df = self.projections[self.projections["State"] == selected_state]
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
        
        @self.app.callback(
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
                state_data = self.data[self.data["State"] == selected_state].iloc[0]
                
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
        
        return self.app

def build_dashboard():
    """Build and return the dashboard application"""
    dashboard = DashboardBuilder()
    app = dashboard.create_layout()
    dashboard.register_callbacks()
    
    return app
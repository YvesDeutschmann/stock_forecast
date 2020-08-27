import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Made with Udacity...", href="https://www.udacity.com/course/data-scientist-nanodegree--nd025")),
            dbc.NavItem(dbc.NavLink("...and Finnhub", href="www.finnhub.io")),
            dbc.NavItem(dbc.NavLink("GitHub", href="https://github.com/YvesDeutschmann/stock_forecast")),
            dbc.NavItem(dbc.NavLink("LinkedIn", href="https://www.linkedin.com/in/yves-deutschmann/")),
        ],
        brand="Stock Market Forecast",
        # brand_href="#",
        color="primary",
        dark=True,
    )

button_group = dbc.Row(
        [
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.DropdownMenu([
                        dbc.DropdownMenuItem("Apple"), 
                        dbc.DropdownMenuItem("Google")],
                    label="Stock Tickers",
                    group=False
                ),
                dbc.Button("1 Day"),
                dbc.Button("7 Day"),
                dbc.Button("14 Day"),
                dbc.Button("28 Day")
                ])
            ], width=12)
            
        ], 
    )


progress = html.Div(
    [
        dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
        dbc.Progress(id="progress"),
    ])

app.layout = dbc.Container(
    [
        navbar,

        button_group,
        html.Div("Placeholder for graph")
        
        
        
        

    # implement progress bar when model is trained

        
], fluid=True)

@app.callback(
    [Output("progress", "value"), Output("progress", "children")],
    [Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    # check progress of some background process, in this example we'll just
    # use n_intervals constrained to be in 0-100
    progress = min(n % 110, 100)
    # only add text after 5% progress to ensure text isn't squashed too much
    return progress, f"{progress} %" if progress >= 5 else ""

if __name__ == "__main__":
    app.run_server(debug=True)
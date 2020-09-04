import os
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px
import pandas as pd

import wrangling_scripts.wrangling as wrangle
from data.get_data import convert_to_timestamp
from wrangling_scripts.symbol import Symbol

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# stock symbols to choose from in app
symbols = {
    'AAPL': 'Apple', 
    'GOOG': 'Google', 
    'MSFT': 'Microsoft', 
    'AMZN': 'Amazon'}

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

button_group = html.Div([
    dcc.Dropdown(
        options= [{'label': item[1], 'value': item[0]} for item in symbols.items()],
        id='ticker_selection',
        placeholder='Select a stock'),
    dcc.DatePickerRange(
        id='timeframe_selection',
        min_date_allowed=datetime(2010, 1, 1),
        max_date_allowed=datetime(2018, 12, 31),
        initial_visible_month=datetime.today(),
        start_date = datetime.today() - pd.Timedelta('5y'),
        end_date= datetime.today()
    ),
    dcc.Dropdown(
        options= [{'label': '{} Day(s)'.format(i), 'value': i} for i in [1, 7, 14, 28]],
        id='forecast_selection',
        placeholder='Select a timeframe to forecast'),
        dbc.Button('Predict', id='button', color='success', block=True)
])

app.layout = dbc.Container(
    [
        navbar,

        dbc.Row([
            dbc.Col([
                button_group
            ], width=2),

            dbc.Col([
                html.Div([dcc.Graph(id= 'graph')])          
            ], width=10)
        ])
    ], fluid=True
)

@app.callback(
    Output('graph', 'figure'),
    [
        Input('ticker_selection', 'value'),
        Input('timeframe_selection', 'start_date'),
        Input('timeframe_selection', 'end_date'),
        Input('forecast_selection', 'value'),
        Input('button', 'n_clicks')]
)
def make_forecast(ticker, start, end, len_forecast, n):
    if n > 0:
        # load data
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        data = wrangle.load_data(symbols.keys(), start, end)

        symbol = Symbol(data, ticker)
        symbol.predict_prices(len_forecast)
        fig = symbol.plot_forecast(len_forecast)
        
        return fig
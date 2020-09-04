import os.path
import itertools
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX
import statsmodels.api as sm
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import plotly.graph_objects as go

class Symbol():
    """
    Creates a class for a stock symbol holding all methods and information related to that symbol.
    """

    def __init__(self, data, name):
        self.name = name
        self.df = data[self.name]
        self.filename = './models/{}.pkl'.format(self.name.lower())
        self.isPickled = True if os.path.isfile(self.filename) else False
        self.preds = None

        self._transform_data(self.df.c)
        
        if self.isPickled:
            self._load_model()
        else:
            self._create_model()
            self._save_model()

    def _transform_data(self, y):
        self.y, self.fitted_lambda = boxcox(y.values, lmbda=None)
        
    def _revert_data_transformation(self, y):
        return inv_boxcox(y, self.fitted_lambda)        

    def _create_model(self):
        """
        Finds the optimal parameter set as input for ARIMA model by comparing AIC & BIC scores.
        
        Arguments:
        y_train - timeseries to fit
        symbol - symbol related to y_train series
        Returns:
        param - tuple (p, d, q, trend) of best parameter set
        """
        # define ranges for GridSearch
        trend = ['c', 't', 'ct']
        range_p = range_q = range(0, 5)
        range_d = range(0, 2)
        # Generate all different combinations of p, d, q and trend triplets
        param_set = list(itertools.product(range_p, range_d, range_q, trend))
        
        results_df = pd.DataFrame(columns=['params', 'aic', 'bic'])

        for param in param_set:
            try:
                mod = SARIMAX(
                    self.y, 
                    order=param[:3], 
                    trend=param[3], 
                    enforce_stationarity=False,
                    enforce_invertibility=False)

                results = mod.fit()

                results_df = results_df.append({
                    'params': param, 
                    'aic': results.aic,
                    'bic': results.bic}, 
                    ignore_index=True)
            except:
                continue
        
        results_df = results_df.sort_values(['aic', 'bic'])
        
        model = SARIMAX(
            self.y, 
            order=results_df.iloc[0].params[:3], 
            trend=results_df.iloc[0].params[3],
            enforce_stationarity=False,
            enforce_invertibility=False)
        self.model = model.fit()

    def predict_prices(self, len_forecast=30):
        """Predict values for the next X business days.
        Arguments:
        len_forecast - integer with number of days to predict
        Returns:
        pred - SARIMAXResults object with predicted values
        """

        self.y_pred = self.model.get_prediction(
            start= self.y.size + 1, 
            end= self.y.size + len_forecast, 
            dynamic=False)

    def plot_forecast(self, len_forecast):
        """
        Plot predicted values vs actual test data.

        Arguments:
        pred - model object with predicted values
        pred_ci - dataframe with data for confidence interval
        len_forecast - integer with number of days to predict
        Returns:
        None
        """
        # transform values back to original scale
        self.forecast = self._revert_data_transformation(self.y_pred.predicted_mean)
        self.y = self._revert_data_transformation(self.y)
        
        # find dates for predicted values
        original_index = self.df.index
        index_pred = pd.bdate_range(start= original_index[-1] + pd.Timedelta('1d'), periods=len_forecast)
        index = original_index.join(index_pred)
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x= index,
                y= self.y,
                name = 'Historical Prices'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = index_pred,
                y = self.forecast,
                name = 'Predicted Values'
            )
        )
        
        fig.update_layout(
            xaxis_rangeslider_visible=True,
            title='{}-Day Forecast for ticker: {}'.format(len_forecast, self.name))
        
        return fig

    def _load_model(self):
        """Loads existing model to make further predictions."""
        self.model = SARIMAXResults.load(self.filename)

    def _save_model(self):
        """Save the model as pickle file to models directory."""
        self.model.save(self.filename)
        

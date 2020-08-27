import sys
import itertools
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX
import statsmodels.api as sm

from wrangling_scripts.forecasting_metrics import mape, mase
from data.get_data import load_data


warnings.filterwarnings('ignore')

def prepare_data(data, symbol, test_size=30):
    """
    Load dataset for a specified symbol from the dictionary.
    Tranform and split the data in train and test set.
    Arguments:
    symbol - symbol from existing dataset in ohlc_data dictionary
    Returns:
    y_train - SQRT transformed train dataset
    y_test - SQRT transformed test dataset
    df - original dataframe with transformed data
    """
    
    # load dataframe from dictionary
    df = data[symbol]
    df['c_transform'] = np.sqrt(df.c)
    y_train = df.c_transform[:-test_size]
    y_test = df.c_transform[-test_size:]

    return y_train, y_test

def fit_optimal_model(y_train):
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
                y_train, order=param[:3], 
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
        y_train, 
        order=results_df.iloc[0].params[:3], 
        trend=results_df.iloc[0].params[3],
        enforce_stationarity=False,
        enforce_invertibility=False)
    fitted_model = model.fit()

    return fitted_model

def predict_prices(model, y_train, y_test, len_forecast):
    """Predict values for the next X business days.
    Arguments:
    y_train - training dataset
    y_test - test dataset
    len_forecast - integer with number of days to predict
    visualization - True if visualization of data is desired
    Returns:
    None
    """

    pred = model.get_prediction(
        start=y_train.size + 1, 
        end=y_train.size + len_forecast, 
        dynamic=False)
    
    return pred

def plot_ohlc(data, symbol):
    """
    Plots the ohlc data for a given symbol and timeframe.
    Arguments:
    symbols - list of strings representing the name of a stock symbol that is contained in the ohlc dictionary.
    Returns:
    None
    """
    
    df = data[symbol]
    
    fig = go.Figure(data=[
        go.Candlestick(
            name=symbol,
            x=df.index,
            open=df.o,
            high=df.h,
            low=df.l,
            close=df.c)])

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title='OHLC Stock Chart for: {}'.format(symbol))

def evaluate_model(pred, y_test, len_forecast):
    """
    Compares predicted values to actual test data.
    Arguments:
    pred - predicted dataset
    y_test - test dataset
    Returns:

    """
    forecast =  pred.predicted_mean.values ** 2
    actual = y_test[:len_forecast].values ** 2

    mape_score = mape(actual, forecast)
    mase_score = mase(actual, forecast)

    return mape_score, mase_score

def save_model(model, symbol):
    """
    Save the model as pickle file to models directory.
    Arguments:
    model - model to save
    symbol - symbol name under which the model will be saved
    Returns:
    None
    """
    filename = ''.join([symbol.lower(),'.', 'pkl'])
    try:
        model.save('../models/{}'.format(filename))
        print('Trained model saved as "{}"'.format(filename))
    except:
        print('Saving model failed!')   

def load_model(symbol):
    """
    Loads existing model to make further predictions.
    Arguments:
    symbol - name of the stock symbol to load the corresponding model
    Returns:
    model - SARIMAXResults object with information of the already fitted model
    """
    try:
        model = SARIMAXResults.load('./models/{}.pkl'.format(symbol.lower()))
    except:
        print('Loading model failed!')
    
    return model

def main():
    if len(sys.argv) > 2:
        
        len_forecast = int(sys.argv[1])
        symbols = list(sys.argv[2:])
        print('Loading data for symbols: {}'.format(symbols))
        data = load_data(symbols)
        
        for symbol in symbols:
            print('Transforming and splitting data...')
            y_train, y_test, = prepare_data(data, symbol ,len_forecast)
        
            print('Searching for optimal model parameters...')
            model = fit_optimal_model(y_train)

            print('Making forecast for {} days.'.format(len_forecast))
            pred = predict_prices(model, y_train, y_test, len_forecast)

            print('Evaluating model...')
            mape_score, mase_score = evaluate_model(pred, y_test, len_forecast)
            print('model performance for {} day(s) forecast:'.format(len_forecast))
            print('MAPE score: {}'.format(mape_score))
            print('MASE score: {}'.format(mase_score))

            print('Saving model to models directory as "{}.pkl"'.format(symbol))
            save_model(model, symbol)
    else:
        print('Please provide  a list of stock symbols as the first argument'\
              ' and the length of the desired forecast as second argument.\n'\
              'Example: python wrangling.py 30 AAPL GOOG')

if __name__ == '__main__':
    main()
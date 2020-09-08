import finnhub
import pandas as pd
from datetime import datetime
from config import finnhub_key

# Setup client (Insert your own key here)
finnhub_client = finnhub.Client(api_key=finnhub_key)

def convert_to_unix(timestamp):
    """Converts a pandas timestamp to a unix integer."""
    return (timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

def convert_to_timestamp(unix):
    """Converts a unix integer into a pandas datetime object."""
    return pd.to_datetime(unix, unit='s')

def load_data(symbols, start, end):
    """
    Queries list of stock symbols for their OHLC data.
    Timeframe is from current day back until the desired timeframe (default: 5 years).
    Start
    Arguments:
    symbols - list of strings containing the stock symbol of the desired stock
    timedelta (optional) - timeframe to query data in the past
    Returns:
    ohlc_data - dict of dataframes containing ohlc data for each symbol over certain timeframe
    """
    ohlc_data = dict()
    # get start & endtime for ohlc data
    start_time = convert_to_unix(start)
    end_time = convert_to_unix(end)
    # set resolution for query to 'Daily'
    resolution = 'D'
        
    # get OHLC data for each symbol
    for symbol in symbols:
        res = finnhub_client.stock_candles(symbol, resolution, start_time, end_time)

        data = pd.DataFrame(res)
        data = data.set_index(convert_to_timestamp(data.t))

        ohlc_data.update({symbol: data})
    
    return ohlc_data
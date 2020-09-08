# Investment and Trading Capstone Project
## Build a Stock Price Indicator

### project overview
Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.

This project uses this historical stock prices from finnhub.io to make predictions on the development of these stocks. The result of this process will implememnted in a website giving the user the possibility to choose a certain timeframe or stock to analyze.

### problem statement
The problem to be tackled in this project is to predict future adjsuted stock closing prices for certain stocks. To do so we will make use of several regression and deep learning models to achieve a maximum of accuracy for our predictions. The user interaction of this project will be implemented in a website/dashboard. There it will be possible to choose the stock of interest and a certain timeframe to predict data for the fututre.
The steps to solve the problem are the following:
 - download and prepare current stock prices for predefined stock symbols
 - train a regression model with the data
 - predict future stock prices
 - evaluate predictions
 - implement training, prediction and visualization in web app


### metrics
The performance of the trained model is measured by the Mean Absolute Percentage Error (MAPE). We use this metric because it is a significant metric regardles of the scale of the series. As it is expressed by a percentage, it is also easier to understand for people with a non technical background. The metric is defined as followed:

$$MAPE = \frac{1}{n} \sum_{t=1}^{n} \left\lvert{\frac{A_t - F_t}{A_t}}\right\rvert * 100$$

where $$A_t$$ is the actual value and $$F_t$$ is the predicted value. 

### data exploration & first implementation
The process of data exploration  and first implememntation can be followed with the notebook "SupportingAnalysis.ipynb" that is part of this repository.

### results
The results we could achieve with our ARIMA model are satisfying with an MAPE < 5% for the investigated data. The implemented GridSearch method to find the optimal parameters seem to work as intended. However there coulf efforts be made to try other models like for example the Long Short Term Memory (LSTM) model.
This accuracy sure isn't good enough to speculate at the stock markets and expect a big margin, but it can be a good starting point for further refinement. Adding additional elements like for example current news about the company could provide more insight for the future development of the so called white noise element of a timeseries of stock prices.

### justification
The application is able to provide reasonable predictions for a limited set of stock tickers within a certain timeframe. As mentioned above there is space for improvements on the algorithm itself by adding technical indicators aswell as on side of usability. A server that provides already fitted models that are always up to date would be a huge improvement over the current state where the user has to wait for every iteration to find the set of otimal parameters to train the model.

### run the program
**Important:** I used the finnhub python API that requires to register at finnhub.io and recieve an free API key to use their service. I strongly advice to use this library over an approach with the requests library as it allows you to make more requests per minute. So before you run this app, head over there, create an account. Now you're required to store this key in an environment variable called 'finnhub_key'. After taking these steps you'll be able to start the app with running the script *run.py*.

# IE 598 Project Assignment 1
## Project Proposal
### Abstract:
The scope of this project is to model and forecast prices levels of different trading instruments such as equity, futures, options etc. We shall restrict our scope to forecast short term prices that can be modeled with the help of either statistical tools or machine learning tools. In this project, we will try out different tools and ensemble of these models to forecast the price levels and compare with the performance of known models used in literature. The subsequent sections focus on literature survey on different models used historically to forecast price levels and the models considered for this project. Other logistics such as project effort distribution and timelines are included at the end.


### Literature Survey:
After surveying the literature, it is evident that price predictions for high frequency trading are a highly complex issue with inconclusive results. Intra-day price predictions are often highly disputed and can rely on countless market indicators. One issue in applying scholarly models to our project is that much of the existing literature is focused on market data with low frequency timestamps. Some sources use indicators like a ten-day moving average to predict price. In terms of High Frequency Trading, HFT, indicators based on day-to-day movements are likely to not be helpful. On the other hand, some sources suggest that price predictors can be used over a variety of time intervals. Therefore, for this project it will be important to survey a variety of predictors using a several indicators and apply them to high frequency data. Some authors suggest the optimal indicators for price prediction change over the course of time and require recurring calculations to determine the best indicators at a given time. Other scholarly work bases price prediction on machine learning concepts like Long Short-Term Memory models and different neural networks.

A summary of literature survey done on different models used to forecast prices are presented below:

1.	Use of ANN(Artificial Neural Network) to predict stock price (FA de Oliveira et al. 2011). This study picks the price window with horizons - 1, 5, 15, 22, 37,44,66, 110 to build a simple ANN and have a good 1-day prediction. Although this work was published when deep learning was slowly gaining popularity, it gives a good understanding of usage of NN to forecast price levels . At this time, we have exposure to more novel ways to deal with using Deep Learning, e.g.  RNN, LSTM, GAN, etc.

2.	There are studies using SVR(support vector regression) to calculate the weighted moving average(WMA) and then use SVM methods to calculate the prediction. They do not perform better than neural networks but they can still predict the overall trend of the stock price.(Henrique et al. 2018)

3.	There are also recent papers consider using NLP methods to perform financial trends. (E.g. Muthukumar et al. 2021, and Savaş Yıldırım 2018, etc.) For example,  Muthukumar use a method called GAN(Generative Adversarial Network), and they have a RMSE(root mean  square error) smaller than other methods (including traditional time series methods like ARIMA/ sentiment Analysis).

4.	Dev Shah, in 2019, give out a survey/ review of the prediction techniques. This is a good resource that helps with how different models perform in predicting stock levels.

5.	X Pang(2020)’ paper has been cited by a lot of other researchers. They use deep long short-term memory neural network(LSTM). The design of their LSTM is quite complex, but from the experiment, their accuracy is about 50%. But as Dev Shah et al. mentioned, LSTM might work better than RNN, and they are used to predict the long term price prediction.

There are two kinds of inputs we are considering for this project:
- Market by Order(MBO) time series data: This data not only provides the price level information but also the full/partial depth of book. Therefore, the model can use this information to also judge if the instrument is oversold or high in demand and try to forecast the price level behaviour. The source of this data is IEX DEEP historical data.
- Historical price level information: This is the closing price on a 1Min time period that can be used to model the behaviur of price level movement. The source of this data is paper trading API from alpaca markets.


### Project considerations

A few algorithms that are being considered for this project are:
- Extended Kalman Filter: 
  - Kalman filter is a state estimation algorithm that is effectively used in control systems to predict the state (ex: trajectory) based on previous output of the system. The same algorithm can also be used in a discrete sense to predict price level curve based on previous price levels. Extended Kalman filter is an extension of this algorithm that can generalize data where the underlying dynamics is non linear.
  - Input Data: Close price of bar API from Alpaca (historical). We will use the maximum frequency data which is 10000 data points/min (every 6ms).
  https://alpaca.markets/docs/broker/market-data/historical/


- Fast Fourier Transform (FFT): 
   - FFT is another data modelling technique that is extensively used in engineering applications such as signal processing. This transformation technique models the price level curve as a combination of signals with different amplitudes and frequencies. Given historical data of any price level, it can output a model that can predict the future price level fairly accurately provided the model is not overfit to the noise in the data.
  - Input Data: Close price of bar API from Alpaca (historical). We will use the maximum frequency data which is 10000 data points/min (every 6ms).
  https://alpaca.markets/docs/broker/market-data/historical/

- Machine learning models: 
    - We shall be trying out several machine learning based models such as SVM regressor, KNN regressor, Decision Tree regressor and its ensembled counterparts such as Adaboost and Random Forest Regressors. These models can make use of market depth unlike the above models and hence MBO data would be the input here. Prior to fitting these models feature extraction (PCA) and engineering (adding percent changes) would be done to fit these models well with regard to time series data
    - Input Data: IEX DEEP Historical data with 1ms frequency

- LSTM: 
    - LSTMs are way of including long term and short term memory in deep networks. This effectively memorizes patterns in the data that lead to a unique effect on the target. Given this behavior, LSTMs would be perfectly suited for price level forecasting.
    - Input Data: Close price of bar API from Alpaca (historical). We will use the maximum frequency data which is 10000 data points/min (every 6ms).
  https://alpaca.markets/docs/broker/market-data/historical/

### Team member roles:

All the team members will be contributing in coding different models and the raw data feed API parsers but some of the major contributions anticipated are listed below:
- Prajwal Chinthoju - Fit Kalman filter and FFT models and contribute to LSTM model
- Xuan Zhang - Contribute in building LSTM model and developing IEX DEEP parser
- Sheng-Yu Lin - Contribute in IEX DEEP parser and machine learning models
- Richie Ma - Contribute in building machine learning models and help with preprocessing
- Adam Szott - Contribute in verification and comparison of results and preprocessing


## Project TimeLine

- [Week 1] - Understand different models and APIs available
- [Week 2] - Code different models on example data and start coding API parsers
- [Week 3] - Preprocessing market data so that it can be used to fit different models
- [Week 4] - Fit different models to market data
- [Week 5] - Fine tune models for different hyperparameters
- [Week 6] - Compare accuracies of different models and build and ensemble model

### References
- De Oliveira, Fagner Andrade, et al. "The use of artificial neural networks in the analysis and prediction of stock prices." 2011 IEEE International Conference on Systems, Man, and Cybernetics. IEEE, 2011.
- Henrique, Bruno Miranda, Vinicius Amorim Sobreiro, and Herbert Kimura. "Stock price prediction using support vector regression on daily and up to the minute prices." The Journal of finance and data science 4.3 (2018): 183-201.
- Muthukumar, Pratyush, and Jie Zhong. "A stochastic time series model for predicting financial trends using nlp." arXiv preprint arXiv:2102.01290 (2021).
- Yıldırım, Savaş, et al. "Classification of" Hot News" for Financial Forecast Using NLP Techniques." 2018 IEEE International Conference on Big Data (Big Data). IEEE, 2018.
- Pang, Xiongwen, et al. "An innovative neural network approach for stock market prediction." The Journal of Supercomputing 76.3 (2020): 2098-2118.
- Shah, Dev, Haruna Isah, and Farhana Zulkernine. "Stock market analysis: A review and taxonomy of prediction techniques." International Journal of Financial Studies 7.2 (2019): 26.

Other Potential Market data Sources:

- NASDAQ ITCH: ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/
- Deutsche Börse Public Data Set: https://registry.opendata.aws/deutsche-boerse-pds/

Other Potential stock market data API source:

- Yahoo Finance API
- Alpha Vantage API
- Stock and Options Trading Data Provider API
- Investing Cryptocurrency Markets API
- Zirra API
- Twelve Data API
- Finage Currency Data Feed API
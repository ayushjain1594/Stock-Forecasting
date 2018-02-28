# Stock-Forecasting
Hidden Markov Model (HMM) based stock forecasting.

NOTE: *Refer Final_Report.pdf for full documentation*

Stock markets are one of the most complex systems which are almost impossible to model in terms of dynamical equations. The main reason is that there are several uncertain parameters like economic conditions, company's policy change, supply and demand between investors, etc. which drive the stock prices. These parameters are constantly varying which makes stock markets very volatile in nature. Prediction of stock prices is classical problem of non-stationary pattern recognition in Machine Learning. There has been a lot of research in predicting the behavior of stocks based on their historical performance using Artificial Intelligence and Machine Learning techniques like- Artificial Neural Networks, Fuzzy logic and Support Vector Regression. One of the methods which is not as common as the above mentioned for analyzing the stock markets is Hidden Markov Models. Hence, we will be focusing on Hidden Markov Models in this project and compare its performance with Support Vector Regression Model.

## Data files
The files contain daily stock prices (ex. google.csv) in order- Close, Open, High, Low.

The output files (forecast) have the predicted prices in the same order for the last 100 days in the training set.

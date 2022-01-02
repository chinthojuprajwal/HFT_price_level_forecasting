# IE 598 PROJECT REPORT: 
# Prediction of short term price level changes using Machine learning
#### Team members and roles:
- Prajwal chinthoju - SVR, Random forest, LSTM and Kalman filter models
- Adam Szott - Rolling regression model
- Richie Ma - Preliminary Analyses
- Xuan Zhang - GAN Model and literature survey
- Sheng-Yu Lin - IEX Data parsing
## Introduction:

This project focuses on prediction of short term changes in price levels using ML and other regression techniques. The problem of price level prediction has been well studied in the literature and is also very challenging given the amount of unpredictability in price movements. In the work that follows, we restrict our models to prediction of short term data which is randomly selected from 1 day's data for SPY ETF prices to minimise the influence of any events on stock prices (for example, an unexpected news event that increases the price in short term). Also, we predict the actual price level change percentage although we don't necessarily get a good accuracy on the price change itself. 

Following is a chart that summarises the contents of this report.
![](pic/hft chart.png)


## Price movement prediction using SVR, Random Forest and LSTM:

The following models only focus on prediction of percentage change of price level(computed as the weighted average of 3-levels of ask and bid prices from book updates) using SVR, random forest and LSTM methods. However since the actual accuracy of the raw price change percentages is very low due to the unpredictable price movements, the final accuracy is as the ratio of correct price movement direction predictions to the total number of book updates.

### Formatting of IEX data
The main objective of this step is to format the IEX data into features and target variables which can then be used in training and testing the algorithm. From the IEX data we use the following fields as the features (X):
- 3 levels of Ask and bid prices of last 20 updates (120 variables)
- 3 levels of Ask and bid sizes of last 20 updates (120 variables)
- timestamps in nanoseconds for last 20 book updates (20 variables)
- Weighted price averages for last 20 book updates (20 variables)
- The timestamp for which the price needs to be predicted (1 varaiable)

Therefore, we have 281 features in total per one training sample

The target variable is the weighted average of price level at the 20+1th timestamp
### Preprocessing of IEX data

The 281 features extracted from the IEX book updates data along with the target variable are all fit to a standard scaler that normalises and standardises the data ( zero mean and unit variance). Also, given that time series data is highly correlated, it presents us with an opportunity to represent the input data in a smaller feature space using PCA. A simple analysis of variance ratio for each principal component suggests that 100 components is enough to explain 90% of the variance, and hence the number of components used in PCA fit is 100 components.

### Support Vector Machine:

This algorithm is mostly used in classification applications but can be extended to regression as well. SVM essentially maximises the gap between two linear separators that classifies maximum number of samples. Another step that is invovled is kernelising the input data, which is basically a transformation of input data into a different space.One such kernel function is Radial Basis Function, which performs a non-linear tranformation of the input space. Using this kernel, we can linearly separate non linear data. We used the following hyperparameters for the  SVR fit:

Model parameters:
- Kernel used: RBF
- Grid search:
- Gamma=[0.0001,0.001,1,10,100]
- C=[10,1,0.1]
- Cross validation split= 5
- Train data set size=7000 samples
- Test data set size=3000 samples
- Best params: C=0.1; Gamma=0.01
- 
Results:
Training accuracy =56.5%
Test accuracy = 49.63%

### Random Forest:

Random forest model uses a collection of Decision trees to accurately classify/predict a given target. Each decision tree is trained on a subset of the entire dataset. The test data sample is then fed to all the decision trees and their results are ensembled using a majority vote or averaged (in case of regression).

Model Parameters:
- Number of estimators used=150
- Max-depth used in grid search={3,4,5,7,10}
- Cross validation split= 5
- Train data set size=7000 samples
- Test data set size=3000 samples
- Best max-depth=3

Model Accuracy (ratio of correct predictions):

- Training accuracy=55.3%
- Test accuracy =51.2%

### Long Short Term Memory:

LSTM is a form of RNN layer that is capable of handling problems that require long term memory. There have been several attempts at using LSTM for stock price prediction because of the property that it is able to identify and recognise patterns in price level movements. We use the following model to train for SPY data:

Model Summary:
- LSTM 256 units, Activation function- tanh
- Dense layer 10 units, activation function- linear
- Dense layer 10 units
- Dense output layer 1 units

Model Accuracy (ratio of correct predictions):

- Training accuracy=50.9%
- Test accuracy =50.1%


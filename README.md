# ANALYZING RETAIL CUSTOMERS AND PREDICTING THEIR BEHAVIOR USING DATA MINING AND MACHINE LEARNING TOOLS (2020 PROJECT)

## Clustering
In the first, we will first aggregate the data from a real department store  in multiple dimensions. 
Then we will build several clustering models and analyse their performance on this real data.
 We will conclude, in what is the best model that can be used for department stores based on their characteristics.

## Classification
In the second part of the study, we will propose different classification models with the aim to predict the customer label at the end of the year based on quarterly data. 
The models will be based on Support Vector Machines (SVM), K-Nearest Neighbours (KNN)  neural networks and XGBoost algorithms.
 The customer label is a number allocated to each customer based on their annual expenses. Or models will be trained and tested in our dataset. 
 Their performance will be evaluated on accuracy percentage metric.

## Time Series
In the third part of our study, we will analyse time series models and particularly propose a modified Prophet model to predict daily sales in the department stores.
 The base prophet model and the modified one will be evaluated based on prediction MAE (Mean Absolute Error). 
 The modified model will be trained and tested in the sales of the department store from 2011-2019 and its performance will be evaluated bases on prediction MAPE (Mean Absolute Percentage Error).


## Status
Clustering, NN and Prophet implementation and optimization are uploaded.
SVM and XGBoost implementation will be uploaded soon.
Results of tuning the neural network and optimizing the Prophet model are uploaded. Clustering results will be uploaded soon.
For the dataset please contact me at led.lico@gmail.com.

## Results

### Neural Network
The model with neural networks that we proposed is very promising and should be researched and experimented more in the future with different model architectures. Maximum accuracy was achieved for the architecture with a hidden layer of 20 neurons and with sigmoid activation function. This conclusion was reached after many experiments with different activation functions and by alternating the number of neurons in the intermediate layer. The accuracy achieved in the prediction can be considered very good if we refer to the number of data used for our study.
### Prophet results
In the third part of the project , models related to time series were analyzed and proposed, with a main focus on the Prophet algorithm. Some changes from the baseline model were proposed which gave us approximately 1.4% correction of the forecast error (MAE). We noticed that the change in the order of the Fourier series in the sales seasonalities gave us the greatest impact on the forecast. The proposed model gave very good results for forecasts up to 90 days (MAPE <25%). In the range of 90-140 days we had an increase in error which stabilized again after this range. This problem is thought to come from the change of promotional seasons in the years 2018/2019 but will be analyzed in more detail and will be addressed in future studies.

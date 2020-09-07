# Time-Series-Forecasting-with-Deep-Learning

The purpose of this project is to explore how Multi-Task Learning(MTL) will improve the prediction accuracy in time series forecasting than Single-Task Learning(STL). 
We use the Temporal Convolution Network(TCN) proposed by (Bai, Kolter and Koltun, 2018). 
In terms of the dataset, two stock market datasets are used to test the model which are SP 500 Index and Shanghai Index. 
We also propose a data preprocessing method for time series predicting based on the idea of residual connection. We are not counting on the residual connection as a model structure but as a way to pre-process our data, details will be discussed later.
Furthermore, we compare the performance between TCN and LSTM.
The focus of this project is to compare Multi-Task Learning and Single-Task Learning and we perform several experiments and draw solid conclusions based on them and find that Multi-Task Learning will help the model avoid overfitting and improve the overall predicting ability.


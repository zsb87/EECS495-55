#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=================================================================
Support Vector Regression for Multivariate Time Series Prediction
=================================================================
@author: Yuanhui Yang
@email: yuanhui.yang@u.northwestern.edu
@reference:
:http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html
:http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html
=================================================================
"""
print(__doc__)

import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt


class Data(object):
	def __init__(self, length_of_sequence):
		self.length_of_sequence = length_of_sequence
		self.sequence = 5 * np.random.rand(self.length_of_sequence)

length_of_sequence = 100
data = Data(length_of_sequence)
data.length_of_unit = 5
data.input = np.zeros((data.length_of_sequence - data.length_of_unit, data.length_of_unit))
data.output = np.zeros((data.length_of_sequence - data.length_of_unit, 1))
for i in range(0, data.length_of_sequence - data.length_of_unit):
	data.output[i] = data.sequence[data.length_of_unit + i]
	for j in range(0, data.length_of_unit):
		data.input[i][j] = data.sequence[i + j]
data.length_of_prediction_sequence = 20
data.input_scaler = preprocessing.StandardScaler()
data.input_transform = data.input_scaler.fit_transform(data.input)
data.output_scaler = preprocessing.StandardScaler()
data.output_transform = data.output_scaler.fit_transform(data.output)
data.input_transform_train = data.input_transform[:(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence)]
data.output_transform_train =  data.output_transform[:(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence)].ravel()
data.input_transform_test = data.input_transform[(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence):]
data.output_test = data.output[(data.length_of_sequence - data.length_of_unit - data.length_of_prediction_sequence):].ravel()
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10, param_grid={"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "C": np.logspace(-10.0, 10.0, num=40, base=2.0), "gamma": np.logspace(-10.0, 10.0, num=40, base=2.0)})
svr.fit(data.input_transform_train, data.output_transform_train)
data.output_transform_predict = svr.predict(data.input_transform_test)
data.output_predict = data.output_scaler.inverse_transform(data.output_transform_predict.reshape((data.length_of_prediction_sequence, 1)))

plt.figure()
x = np.arange(0, data.length_of_prediction_sequence)
y1 = data.output_test
y2 = data.output_predict
plt.plot(x, y1, 'ro-', linewidth=2.0)
plt.plot(x, y2, 'bo-', linewidth=2.0)
plt.grid(True)
plt.show()

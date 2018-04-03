# JAscripts
# Boston
# Jasser Alshehri
# Starkey Hearing Technologies
# 3/17/2018

# Taken from https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import sklearn

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

from matplotlib import rcParams

from sklearn.datasets import load_boston

boston = load_boston()              # import Boston housing dataset
bos = pd.DataFrame(boston.data)     # reformat for pandas data manipulation
bos['PRICE'] = boston.target        # bring in the "price" column to the data set
X = bos.drop('PRICE', axis = 1)     # define X (features) as the input, which excludes the "Price" column
Y = bos['PRICE']                    # define Y as the output based as it relates to input X. This ouptut is "Price"

from sklearn import model_selection # for some reason, sklearn.model_selection won't work, so explicitly called it out here
# The following command splits the dataset into 2 segments, one of which is to train the algorithm, the other set is to test it.
# The amount of each segment is based on "test_size" parameter.
# not sure what "random_state" parameter does here.
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)            # Defines lm as Linear Regression, and fitting it to the training data set

Y_pred = lm.predict(X_test)         # Defines Y_pred as the Linear Regression output based on the X_test data set.

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)    # Calculate Mean-Squared-Error between predicted and actual values of "Price"
print "mse: " + str(mse)

plt.figure()
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

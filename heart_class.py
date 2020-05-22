# import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import svm

dataR = pd.read_csv("winequality-red.csv", sep = ";")
dataW = pd.read_csv("winequality-white.csv", sep = ";")

data = dataR.append(dataW)

print("All data: ")
print(data)


# Get colume
# data["columeName"]

# Get single row
# dataRow = data.loc[0]
# print(dataRow)

dataSet = data

#dataSet = data[["Age","Sex","CP","TestsBPS","COL","FBS","HRrest","HRmax","Exang","OldPeak","Slope","CEA","TAL","NUM"]]

print("Dataset is: ")
print(dataSet)

predict = "quality"

x = np.array(dataSet.drop([predict], 1))
y = np.array(dataSet[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

# regr = svm.SVR()

print("x_train, with size {}: ".format(len(x_train)))
print(x_train)

print("y_train, with size {}: ".format(len(y_train)))
print(y_train)

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

# regr.fit(x_train, y_train)
# acc = regr.score(x_test, y_test)
print("Accuracy is {}!".format(acc))
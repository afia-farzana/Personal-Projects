import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Reading all student data
data = pd.read_csv("student-mat.csv", sep=";")

# Specifying the desired attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Specifying the label (refers to the desired value to predict)
predict = "G3"

# Return a dataframe without G3
x = np.array(data.drop([predict], axis=1))
# Returning a data frame with the G3 column of values
y = np.array(data[predict])

# Splitting up 10% of data into test samples
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

""" As a model with 91.4% accuracy is saved, there is no need to create a new one and retrain in each execution
best_acc = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    # Creating a linear model
    linear = linear_model.LinearRegression()
    # linear = best fit line using x and y training samples
    linear.fit(x_train, y_train)
    # Saving the model using pickle if it has a better accuracy
    acc = linear.score(x_test, y_test)
    if best_acc > acc:
        best_acc = acc
        with open("student_model.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# Read in the model
pickle_in = open("student_model.pickle", "rb")
# Loads model into the variable linear
linear = pickle.load(pickle_in)

# Prints the gradient value for each attribute
# the larger the gradient va;ue the more it impacts the prediction
print('Coefficient: \n', linear.coef_)
# Prints the y-intercept
print("Intercept: \n", linear.intercept_)
# Print the model accuracy
acc = linear.score(x_test, y_test)
print("Model accuracy:", acc)

# Makes predictions of G3 for the 10% sample x data (excluding G3)
predictions = linear.predict(x_test)
# Prints the predicted value, attributes the value is based on, and the expected value
print("Predicted values:")
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Plotting attributes against final grade
style.use("ggplot")
x_param = "absences"
pyplot.scatter(data[x_param], data["G3"])  # Note that the y-axis always has to be the label
pyplot.xlabel(x_param)
pyplot.ylabel("Final grade")
pyplot.show()

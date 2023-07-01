import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from matplotlib import style
import matplotlib.pyplot as pyplot

# Reading in the data
data = pd.read_csv("car.data")

# An object to encode values (e.g. high, medium, low) into integers
le = preprocessing.LabelEncoder()
# Convert columns into numpy arrays and encode non-numeric values into integers
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class_attribute = le.fit_transform(list(data["class"]))

# Splitting the training and testing sets
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(class_attribute)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Creating a set of K values and corresponding accuracy
k_val = []
model_acc_plot = []
for _ in range(1, 51, 2):
    # Appending k value
    k_val.append(_)
    # Creating and training the model
    model = KNeighborsClassifier(n_neighbors=_)
    model.fit(x_train, y_train)
    # Appending accuracy value
    model_acc_plot.append(model.score(x_test, y_test))
# Mapping accuracy against increasing numbers of neighbours
style.use("ggplot")
pyplot.scatter(k_val, model_acc_plot)  # Note that the y-axis always has to be the label
pyplot.xlabel("K Values")
pyplot.ylabel("Model Accuracy")
pyplot.show()

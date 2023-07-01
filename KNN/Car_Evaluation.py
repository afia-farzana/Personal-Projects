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

# Creating and training the KNN model
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)

# Testing model accuracy
model_acc = model.score(x_test, y_test)
print(model_acc)

# Predicts the classification
predicted = model.predict(x_test)

# Print the predicted value, data used to make the prediction, and the actual value
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # Find the 9 nearest neighbours to the point
    n = model.kneighbors([x_test[x]], 9, True)
    # Print their distance from the point and their index
    print("N: ", n)



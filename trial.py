import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())  # PRINTS THE DATA ACCORDING TO THE COLUMN NAME GIVEN IN THE LINE:8

predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# SO X_TEST HAS ALL THE VALUES EXCEPT G1 AND Y_TEST HAS ALL THE VALUES OF G1

"""best = 0
for _ in range(10000):
    linear = linear_model.LinearRegression()  # CREATING A MODEL FOR THE DATA
    linear.fit(x_train, y_train)  # GIVES THE BEST LINE THE SCATTERED DATA CAN MAKE
    acc = linear.score(x_test, y_test)  # FINDING THE ACCURACY OF THE SCORE
    print(acc)

    if acc > best:
        best = acc
    with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f)"""


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


predictions = linear.predict(x_test)  # PREDICTS WHAT G1 WOULD BE ON BASIS OF THE OTHER THINGS
for item in range(len(predictions)):
    print(predictions[item], x_test[item], y_test[item])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])  # GIVING X AND Y VALUES
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()  # IN THE GRAPH YOU CAN SEE THE DIFFERENT TYPE OF RELATION BY CHANGING THE VALUE OF p AND SEE WHICH
# VALUES AFFECT THE LINEARITY OF G3

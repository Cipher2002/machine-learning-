# SO BASICALLY SVM (SOFT VECTOR MACHINE) IS USED TO ORDER THE UNORDERED DATA AS IN CONVERT THE 2D DATA INTO 3D DATA
# SO THAT THE DATA THAT IS IN LINE/BETWEEN EACH OTHER MAKE THEM SHOOT UP SHOOT DOWN SO THAT ALL THE DIFFERENT
# TYPE OF DATA IS TOGETHER.

# SOFT MARGIN IS A PARAMETER IS A PARAMETER THAT WE SET THAT IS ALLOWED TO BE IN THE SOFT MARGIN THAT LIE BETWEEN THE
# TWO POINTS CLOSEST TO THE HYPERPLANE. THIS ALLOWS THE DATA FROM DIFFERENT GROUP BE WITH ANOTHER GROUP

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

""""print(cancer.feature_names)"""  # I BELIEVE THIS IS THE DATASET CONTAINING ALL THE INFORMATION FOR SOMETHING TO BE
# RECOGNIZED AS CANCER
"""print(cancer.target_names)"""  # AGAIN I BELIEVE THIS IS THE OUTCOME THAT WE WANT AFTER PREDICTING USING THE DATASET

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train, y_train)
classes = ['malignant' 'benign']

# clf = KNeighborsClassifier(n_neighbors=13)  # THIS IS THE "K NEAREST NEIGHBORS (KNN)" ML ALGO
# clf = svm.SVC(kernel="linear")  # THIS IS THE "SOFT VECTOR MACHINE (SVM)" ML ALGO

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

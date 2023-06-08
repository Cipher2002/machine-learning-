# SO THE K MEANS ALGO WE DONT HAVE TO TELL IT WHAT THE TEST AND TRIAL DATA IS. IT BASICALLY FINDS K NUMBER OF CENTROID
# IN THE DIFFERENT GROUPS OF DATA AND THEN AFTER CONTINUOUS ITERATIONS IT FINDS THE BEST PLACE TO KEEP THOSE CENTROID
# AND DIFFERENTIATES THE GROUP SO THAT WHEN WE ASK IT WHERE A POINT IS IT CAN TELL WITH ACCURACY THAT IT FALLS WITHIN
# THIS GROUP

import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)
y = digits.target

k = 10

samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

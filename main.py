# Lucas Weakland
# SCS 235
# Final Project

# imports
# I know theres a lot of imports, this was A LOT of testing. Thought id keep it in hahaha
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from random import sample
import math
from sklearn.datasets import load_iris
from sklearn import datasets, linear_model, metrics
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# load data set
dataset = load_iris()

# !!! Binning & Sampling & Smoothing !!!
a = dataset.data
b = np.zeros(150)

# take 1st column among 4 column of data set (which i stole bin1 and used that)
for i in range(150):
    b[i] = a[i, 1]

b = np.sort(b)  # sort the array

# create bins (this is also our sample)
bin1 = np.zeros((30, 5))


# Bin mean
for i in range(0, 150, 5):
    k = int(i / 5)
    mean = (b[i] + b[i + 1] + b[i + 2] + b[i + 3] + b[i + 4]) / 5
    for j in range(5):
        bin1[k, j] = mean
print("Binning Data... \n", bin1)

# !!! Normalization !!!
# SIDE NOTE PLEASE READ - So i did this, you can un comment it out, but it works. I just don't know if its right
# so im keeping it commented-out for now. Ty!
#c = a*20

# normalize the data attributes
#normalized = preprocessing.normalize(bin1)
#print("Normalized Data = ", normalized)

# !!! Bayes Theory !!!
# store the feature matrix (X) and response vector (y)
X = dataset.data
y = dataset.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# training the model on training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
print("Bayes model accuracy (in %): ", metrics.accuracy_score(y_test, y_pred) * 100)

# !!! Regression !!!

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Regression : Coefficients: ', reg.coef_)

# variance score
print('Regression : Variance score: {}'.format(reg.score(X_test, y_test)))


# setting plot style
plt.style.use('fivethirtyeight')

# plotting errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# plotting errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# line for 'zero residual error'
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

# line for upper right
plt.legend(loc='upper right')

#  title
plt.title("Residual errors")

# function to show plot
plt.show()

# !!! Euclidean / Manhattan Distance !!!
# I used Euclidean

# intializing points in numpy arrays
eucPoint = np.array(bin1)

# calculating Euclidean distance
dist = np.linalg.norm(eucPoint)

# printing Euclidean distance
print("Euclidean Distance: ", dist)

# !!! Dimensionality Reduction !!!

# data
X = np.array(bin1)

# creating an instance
kernel_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.03)

# fit and transform the data
final = kernel_pca.fit_transform(X)

# prints a 2d-array
print("Dimensionality Reduction: \n", final)

# !!! Clustering !!!

# making scatter (elbow method)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

# titling
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# showing cluster
plt.show()

# non elbow way, or just normal scatter / cluster
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# showing cluster
plt.show()
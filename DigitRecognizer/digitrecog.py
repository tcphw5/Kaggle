# -*- coding: utf-8 -*-
"""
Kaggle Competition #2 Digit Recognizer

Created on Sun Apr 23 11:43:02 2017

@author: tpwin10
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")

target = traindf['label']
traindf = traindf.drop("label", axis=1)

pca = decomposition.PCA(n_components=200)
pca.fit(traindf)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% of var explained')

pca = decomposition.PCA(n_components=100)
pca.fit(traindf)
PCtrain = pd.DataFrame(pca.transform(traindf))
PCtrain['label'] = target

pca.fit(testdf)
PCtest = pd.DataFrame(pca.transform(testdf))
      
y = PCtrain['label'][0:20000]
X=PCtrain.drop('label', axis=1)[0:20000]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2500,), random_state=1)
clf.fit(X, y)

predicted = clf.predict(PCtrain.drop('label', axis=1)[20001:420000])
expected = PCtrain['label'][20001:42000]

output = pd.DataFrame(clf.predict(PCtest), columns = ['Label'])
output.reset_index(inplace=True)
output.rename(columns={'index': 'ImageId'}, inplace=True)
output['ImageId'] = output['ImageId']+1
output.to_csv('output.csv', index=False)
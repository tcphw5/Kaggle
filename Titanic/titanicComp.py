# -*- coding: utf-8 -*-
"""
Kaggle Competition #1 Titantic Data Set

Created on Sat Apr 22 17:23:56 2017

@author: tpwin10
"""

import pandas as pd
import numpy as np
import re
import operator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def get_family_id(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(),
                              key=operator.itemgetter(1))[1] + 1)
            family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

family_id_mapping = {}

training_df = pd.read_csv("train.csv")
testing_df = pd.read_csv("test.csv")

training_df["Age"] = training_df["Age"].fillna(training_df["Age"].median())
training_df["Fare"] = training_df["Fare"].fillna(training_df["Fare"].median())

training_df.loc[training_df["Sex"] == "male", "Sex"] = 0
training_df.loc[training_df["Sex"] == "female", "Sex"] = 1

training_df["Embarked"] = training_df["Embarked"].fillna("S")
               
training_df.loc[training_df["Embarked"] == "S", "Embarked"] = 0
training_df.loc[training_df["Embarked"] == "C", "Embarked"] = 1
training_df.loc[training_df["Embarked"] == "Q", "Embarked"] = 2
 
training_df["FamilySize"] = training_df["SibSp"] + training_df["Parch"]
training_df["NameLength"] = training_df["Name"].apply(lambda x: len(x))
             
#family_ids = training_df.apply(get_family_id, axis=1)
#
#family_ids[training_df["FamilySize"] < 3] = -1
#
#training_df["FamilyId"] = family_ids

titles = training_df["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Dona":9, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}       

for k,v in title_mapping.items():
    titles[titles==k] = v

training_df["Title"] = titles
       
testing_df["Age"] = testing_df["Age"].fillna(testing_df["Age"].median())               
testing_df["Fare"] = testing_df["Fare"].fillna(testing_df["Fare"].median())

testing_df.loc[testing_df["Sex"] == "male", "Sex"] = 0
testing_df.loc[testing_df["Sex"] == "female", "Sex"] = 1

testing_df["Embarked"] = testing_df["Embarked"].fillna("S")
testing_df.loc[testing_df["Embarked"] == "S", "Embarked"] = 0
testing_df.loc[testing_df["Embarked"] == "C", "Embarked"] = 1
testing_df.loc[testing_df["Embarked"] == "Q", "Embarked"] = 2           

testing_df["FamilySize"] = testing_df["SibSp"] + testing_df["Parch"]
testing_df["NameLength"] = testing_df["Name"].apply(lambda x: len(x))              

#family_ids = testing_df.apply(get_family_id, axis=1)
#
#family_ids[testing_df["FamilySize"] < 3] = -1
#
#testing_df["FamilyId"] = family_ids

titles = testing_df["Name"].apply(get_title)

for k,v in title_mapping.items():
    titles[titles==k] = v

testing_df["Title"] = titles

          
          
#alg = LinearRegression()
alg = LogisticRegression(random_state=1)
alg2 = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
                   
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]

selector = SelectKBest(f_classif, k=5)
selector.fit(training_df[predictors], training_df["Survived"])

scores = -np.log10(selector.pvalues_)

algs = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

kf = cross_validation.KFold(training_df.shape[0], n_folds=4, random_state=1)

full_predictions = []
flpr2 = []

for alg, predictors in algs:
    alg.fit(training_df[predictors], training_df["Survived"])
    predictionsTest = alg.predict_proba(training_df[predictors].astype(float))[:,1]
    predictions = alg.predict_proba(testing_df[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
    flpr2.append(predictionsTest)
    
    
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)

predictionsTest = (flpr2[0] * 3 + flpr2[1]) / 4
predictionsTest[predictionsTest <= .5] = 0
predictionsTest[predictionsTest > .5] = 1
predictionsTest = predictionsTest.astype(int)


print(predictionsTest == training_df["Survived"])

counter = 0

for x in range(len(predictionsTest)):
    if predictionsTest[x] == training_df["Survived"][x]:
        counter += 1
        
print(counter/len(predictionsTest))

submission = pd.DataFrame({
        "PassengerId": testing_df["PassengerId"],
        "Survived": predictions})


##########Got a score of .779 on kaggle            

#scores = cross_validation.cross_val_score(alg2, training_df[predictors], training_df["Survived"], cv=kf)


print(scores.mean())

#alg.fit(training_df[predictors], training_df["Survived"])

#predictions = alg.predict(testing_df[predictors])


submission.to_csv("kaggle.csv", index=False)
#print(training_df.head(5))
#print(training_df.describe())

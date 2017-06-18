import pandas as pd
import numpy as py
from sklearn import tree
from sklearn import ensemble
import re


data = pd.read_csv("Titanic Machine Learning from Disaster\\train.csv")
del data["PassengerId"]

names = data["Name"]
title = pd.Series([re.split(" ", name)[1] for name in names])
title_to_titleNum = {
    "Mr.": 0,
    "Miss.": 1,
    "Mrs.": 2
}
title = title.map(title_to_titleNum).fillna(3).astype(int)
del data["Name"]
data.insert(1, "Title", title)

ages = data["Age"]
avage = data["Age"].groupby(data["Title"]).mean().astype(int)
avages = data["Title"].map(avage)
data["Age"].fillna(avages, inplace=True)

data["Sex"].replace(["male", "female"], [0, 1], inplace=True)
data["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)

del data["Ticket"]
del data["Cabin"]

data.dropna(inplace=True)

target = data["Survived"]
del data["Survived"]

traintarget = target[0:800]
traindata = data[0:800]
testtarget = list(target[800:])
testdata = data[800:]

maxlen = 0
best_n_estimators = 0
best_min_samples_leaf = 0
for i in range(100):
    for j in range(30):
        clf = ensemble.RandomForestClassifier(n_estimators=i+20, min_samples_leaf=j+1)
        clf = clf.fit(traindata, traintarget)
        res = clf.predict(testdata)
        final = pd.DataFrame()
        final.insert(0, "R", pd.Series(testtarget))
        final.insert(1, "P", pd.Series(res))
        r = final[final["R"] == final["P"]]
        l = len(r)
        if(l > maxlen):
            maxlen = l
            best_n_estimators = i+20
            best_min_samples_leaf = j+1

"""test = pd.read_csv("Titanic Machine Learning from Disaster\\test.csv")
passengerId = test["PassengerId"]
del test["PassengerId"]

names = test["Name"]
title = pd.Series([re.split(" ", name)[1] for name in names])
title = title.map(title_to_titleNum).fillna(3).astype(int)
del test["Name"]
test.insert(0, "Title", title)

ages = test["Age"]
avage = test["Age"].groupby(test["Title"]).mean().astype(int)
avages = test["Title"].map(avage)
test["Age"].fillna(avages, inplace=True)

test["Sex"].replace(["male", "female"], [0, 1], inplace=True)
test["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)

del test["Ticket"]
del test["Cabin"]

test.fillna(0, inplace=True)

res = clf.predict(test)

final = pd.DataFrame()
final.insert(0, "PassengerId", passengerId)
final.insert(1, "Survived", res)

final.to_csv("Titanic Machine Learning from Disaster\\res.csv", index=False)"""

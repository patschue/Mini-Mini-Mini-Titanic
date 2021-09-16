#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:30:36 2021

@author: Patrick
"""
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

# Laden der Daten
Titanic = pd.read_csv("minichallenge_titanic.csv")
Titanic["ChildAdult"] = np.where(Titanic['Age'] <= 18, "0", "1")


# Transformation der Variable Geschlecht
labelenc = preprocessing.LabelEncoder()
labelenc.fit(Titanic.Sex)
Titanic['Sex'] = labelenc.transform(Titanic.Sex)

# Auswahl von Daten zum Training
train = Titanic.loc[:,['ChildAdult', 'Pclass', 'Sex', 'Survived']].dropna()

# Aufteilen der Daten in Features und Target
y = train['Survived']
X = train.drop('Survived', axis=1)

# Aufteilen in Test und Train Daten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10, shuffle = True)

# Initialisierung des Entscheidungsbaums
clf = DecisionTreeClassifier(max_depth=4)

# Training des Entscheidungsbaums
clf.fit(X_train, y_train)

# Berechnung der Metrik
clf.score(X_test, y_test)

# tree.plot_tree(clf, feature_names = ['Age', 'Fare', 'sex_transform', 'Survived'])
plt.figure(figsize=(40,20))
tree.plot_tree(clf, feature_names = ['ChildAdult', 'Pclass', 'Sex'], fontsize=23)
plt.show()





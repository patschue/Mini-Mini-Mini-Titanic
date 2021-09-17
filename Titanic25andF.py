#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:41:32 2021

@author: schuermi
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
# import graphviz

# Laden der Daten
Titanic = pd.read_csv("minichallenge_titanic.csv")
Titanic["ChildAdult"] = np.where(Titanic['Age'] <= 18, "0", "1")

Titanic25 = Titanic.loc[Titanic['Age'] == 25]
grouped_Titanic25 = Titanic25.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)

TitanicFemale = Titanic.loc[Titanic['Sex'] == "female"]
grouped_TitanicFemale = TitanicFemale.groupby(["Survived", "Pclass"]).size().unstack(fill_value=0)

# TitanicFemale = Titanic.loc[Titanic['Sex'] == "female"]
grouped_TitanicFM = Titanic.groupby(["Survived", "Pclass", "Sex"]).size().unstack(fill_value=0)
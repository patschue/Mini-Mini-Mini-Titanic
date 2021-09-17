#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:58:56 2021

@author: Patrick
"""
import pandas as pd
import numpy as np
# from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns


Titanic = pd.read_csv("minichallenge_titanic.csv")

TitanicDescribe = Titanic.describe()
TitanicType = Titanic.dtypes
TitanicSurvived = Titanic["Survived"]
TitanicFemale = Titanic["Sex"]

Titanic["ChildAdult"] = np.where(Titanic['Age'] <= 18, "Child", "Adult")
Titanic["NoFamilymembers"] = Titanic["SibSp"] + Titanic["Parch"]

grouped_Titanic = Titanic.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
grouped_Titanic.plot.bar()

Titanic.hist(column="Age")
Titanic.hist(column="Fare")
Titanic.boxplot(column='Fare', by=["Survived"])
Titanic.boxplot(column='Age', by=["Survived"])
Titanic.boxplot(column='NoFamilymembers', by=["Survived"])

######################
titanic_db = Titanic
survived = titanic_db['Survived']
pclass = titanic_db['Pclass']
sibsp = titanic_db['SibSp']

sex = {'male': 0, 'female': 1}
embarked = {'S': 0, 'C': 1, 'Q': 2}
child_adult = {'child': 0, 'adult': 1}

titanic_db['ChildAdult'] = np.where(titanic_db['Age'] <= 18, 'child', 'adult')
titanic_db['NoFamilyMembers'] = titanic_db['SibSp'] + titanic_db['Parch']

# Replace Strings in columns with integers to enable plot
titanic_db.replace({'Survived': survived, 'ChildAdult': child_adult, 'Sex': sex, 'Embarked': embarked}, inplace=True)

print(titanic_db.groupby('Sex')[['Survived']].mean())


# Visualize the count of survivors for columns 'ChildAdult', 'Sex', 'Pclass', 'SubSp', 'Parch', and 'Embarked'
cols = ['ChildAdult', 'Sex', 'Pclass', 'NoFamilyMembers', 'Parch', 'Embarked']

n_rows = 2
n_cols = 3

# The subplot grid and the figure size of each graph
# This returns a Figure (fig) and an Axes Object (axs)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):  
        
        # Index to go through the number of columns
        i = r*n_cols + c  
        # Show where to position each subplot
        ax = axs[r][c]
        sns.countplot(titanic_db[cols[i]], hue=titanic_db["Survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right') 
      
# tight_layout
plt.tight_layout() 

titanic_db.hist(column="Pclass")
titanic_db.hist(column="Sex")
titanic_db.hist(column="ChildAdult")
titanic_db.hist(column="Embarked")
titanic_db.hist(column="Survived")

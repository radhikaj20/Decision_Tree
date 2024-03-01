# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:16:58 2024

@author: Acer
"""

"""
Problem statement
2.	 Divide the diabetes data into train and test datasets and 
build a Random Forest and Decision Tree model with Outcome as 
the output variable. 
"""


############# EDA ##########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('C:/12-Decision_Tree/Diabetes.csv')
df.head()
'''
 Number of times pregnant  ...   Class variable
0                          6  ...              YES
1                          1  ...               NO
2                          8  ...              YES
3                          1  ...               NO
4                          0  ...              YES

[5 rows x 9 columns]

'''

######################################################################
df.shape
#(768, 9)

#####################################################################

df.columns

'''
Index([' Number of times pregnant', ' Plasma glucose concentration',
       ' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin', ' Body mass index',
       ' Diabetes pedigree function', ' Age (years)', ' Class variable'],
      dtype='object')
'''

################################################################

df.dtypes

'''
Number of times pregnant          int64
 Plasma glucose concentration      int64
 Diastolic blood pressure          int64
 Triceps skin fold thickness       int64
 2-Hour serum insulin              int64
 Body mass index                 float64
 Diabetes pedigree function      float64
 Age (years)                       int64
 Class variable                   object
dtype: object

'''
##############################################################

df.describe

'''
<bound method NDFrame.describe of       Number of times pregnant  ...   Class variable
0                            6  ...              YES
1                            1  ...               NO
2                            8  ...              YES
3                            1  ...               NO
4                            0  ...              YES
..                         ...  ...              ...
763                         10  ...               NO
764                          2  ...               NO
765                          5  ...               NO
766                          1  ...              YES
767                          1  ...               NO

[768 rows x 9 columns]>
'''

##################################################################
df.info

'''
<bound method DataFrame.info of       Number of times pregnant  ...   Class variable
0                            6  ...              YES
1                            1  ...               NO
2                            8  ...              YES
3                            1  ...               NO
4                            0  ...              YES
..                         ...  ...              ...
763                         10  ...               NO
764                          2  ...               NO
765                          5  ...               NO
766                          1  ...              YES
767                          1  ...               NO

[768 rows x 9 columns]>
'''
####################################################################
df.isnull().sum()

'''
Number of times pregnant        0
 Plasma glucose concentration    0
 Diastolic blood pressure        0
 Triceps skin fold thickness     0
 2-Hour serum insulin            0
 Body mass index                 0
 Diabetes pedigree function      0
 Age (years)                     0
 Class variable                  0
dtype: int64

'''

####################################################################

#The above dataset does not contain any null value
#So no need of data handling

############################################################

from sklearn.preprocessing import LabelEncoder

#Converting into binary
lb = LabelEncoder()
df["Class variable"] = lb.fit_transform(df["Class variable"])

#########################################################33
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('C:/12-Decision_Tree/Diabetes.csv')
data
data.columns

# Separate features (X) and target variable (y)
X = data.drop(' Class variable', axis=1)  # Features
y = data[' Class variable']  # Target variable
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)

model.fit(X_train,y_train)
model.score(X_test,y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


















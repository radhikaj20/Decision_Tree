# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:37:20 2024

@author: Acer
"""

"""
problem statement 
1.	A cloth manufacturing company is interested to know about 
the different attributes contributing to high sales. Build a 
decision tree & random forest model with Sales as target 
variable (first convert it into categorical variable).

"""

############# EDA ##########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/12-Decision_Tree/Company_Data.csv')
df.head()
'''
Sales  CompPrice  Income  Advertising  ...  Age  Education Urban   US
0   9.50        138      73           11  ...   42         17   Yes  Yes
1  11.22        111      48           16  ...   65         10   Yes  Yes
2  10.06        113      35           10  ...   59         12   Yes  Yes
3   7.40        117     100            4  ...   55         14   Yes  Yes
4   4.15        141      64            3  ...   38         13   Yes   No

[5 rows x 11 columns]
'''
#############################################
df.shape
# (400, 11)

df.columns
'''
Index(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US'],
      dtype='object')
'''

#############################################################

df.dtypes
'''
Sales          float64
CompPrice        int64
Income           int64
Advertising      int64
Population       int64
Price            int64
ShelveLoc       object
Age              int64
Education        int64
Urban           object
US              object
dtype: object
'''

#################################################################
df.describe()
'''
 Sales   CompPrice      Income  ...       Price         Age   Education
count  400.000000  400.000000  400.000000  ...  400.000000  400.000000  400.000000
mean     7.496325  124.975000   68.657500  ...  115.795000   53.322500   13.900000
std      2.824115   15.334512   27.986037  ...   23.676664   16.200297    2.620528
min      0.000000   77.000000   21.000000  ...   24.000000   25.000000   10.000000
25%      5.390000  115.000000   42.750000  ...  100.000000   39.750000   12.000000
50%      7.490000  125.000000   69.000000  ...  117.000000   54.500000   14.000000
75%      9.320000  135.000000   91.000000  ...  131.000000   66.000000   16.000000
max     16.270000  175.000000  120.000000  ...  191.000000   80.000000   18.000000
'''

########################################################
df.info

'''
<bound method DataFrame.info of      Sales  CompPrice  Income  Advertising  ...  Age  Education Urban   US
0     9.50        138      73           11  ...   42         17   Yes  Yes
1    11.22        111      48           16  ...   65         10   Yes  Yes
2    10.06        113      35           10  ...   59         12   Yes  Yes
3     7.40        117     100            4  ...   55         14   Yes  Yes
4     4.15        141      64            3  ...   38         13   Yes   No
..     ...        ...     ...          ...  ...  ...        ...   ...  ...
395  12.57        138     108           17  ...   33         14   Yes  Yes
396   6.14        139      23            3  ...   55         11    No  Yes
397   7.41        162      26           12  ...   40         18   Yes  Yes
398   5.94        100      79            7  ...   50         12   Yes  Yes
399   9.71        134      37            0  ...   49         16   Yes  Yes
'''
#################################################################

df.isnull()

'''
Sales  CompPrice  Income  Advertising  ...    Age  Education  Urban     US
0    False      False   False        False  ...  False      False  False  False
1    False      False   False        False  ...  False      False  False  False
2    False      False   False        False  ...  False      False  False  False
3    False      False   False        False  ...  False      False  False  False
4    False      False   False        False  ...  False      False  False  False
..     ...        ...     ...          ...  ...    ...        ...    ...    ...
395  False      False   False        False  ...  False      False  False  False
396  False      False   False        False  ...  False      False  False  False
397  False      False   False        False  ...  False      False  False  False
398  False      False   False        False  ...  False      False  False  False
399  False      False   False        False  ...  False      False  False  False
'''
#The above dataset does not contain any null value
#So no need of data handling

############################################################

from sklearn.preprocessing import LabelEncoder

#Converting into binary
lb = LabelEncoder()
df["ShelveLoc"] = lb.fit_transform(df["ShelveLoc"])
df["Sales"] = lb.fit_transform(df["Sales"])

#important
df['Sales'].unique()
df['Sales'].value_counts
colnames = list(df.columns)

#############################################################

predictors = colnames[:10]
target = colnames[0]

#Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

#help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])
preds=model.predict(test[predictors])
preds
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['predictions'])
np.mean(preds==test[target])

#testing
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target],preds_test,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_test==test[target])

#testing
#now let us check the accuracy on training dataset
preds_train=model.predict(train[predictors])
preds_train
pd.crosstab(train[target],preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])

#######################################################################################

"""A cloth manufacturing company is interested to know about the different attributes 
contributing to high sales. Build a random forest model with Sales as target
variable (first convert it into categorical variable).
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:/1-Python/2-dataset/Company_Data.csv.xls")
df

df.head()

df.info()
#dataset does not contains any null values

#we have to convert sales column to categorical data as it is our target column
df['sales_category'] = 'average'
df.loc[df['Sales']<7,'sales_category'] = 'low'
df.loc[df['Sales']>12,'sales_category'] = 'good'

'''df.describe()
#data is widely distriduted so we have to normalize it 
#normalization 
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(df)
df_norm'''

df['ShelveLoc'].unique()
df['ShelveLoc'] = pd.factorize(df.ShelveLoc)[0]
df['Urban'] = pd.factorize(df.Urban)[0]
df['US'] = pd.factorize(df.US)[0]
df = df.drop('Sales',axis=1)

X = df.drop('sales_category',axis=1)
Y = df.sales_category

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model  = RandomForestClassifier(n_estimators=20)
model.fit(x_train,y_train)


model.score(x_test,y_test)
"""accuracy of model :- 0.8125"""
y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
"""
array([[27,  0,  6],
       [ 4,  0,  0],
       [ 5,  0, 38]], dtype=int64)"""





###########################################################################









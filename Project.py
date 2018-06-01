# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:00:40 2018

@author: Milind-PC
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Korea data.csv')

#Checking dataset info
print(dataset.info())

#data missing for Outlook and Japanese Tourists
sns.heatmap(dataset.isnull())

#Plot for outlook
dataset['Outlook'].unique()

# 'rainy' to 'Rainy' and 'cloudy' to 'Cloudy'
dataset.Outlook.replace(['rainy', 'cloudy'], ['Rainy', 'Cloudy'], inplace=True)

X_outlook = dataset.iloc[:, -3].values

X_outlook = pd.DataFrame(X_outlook)

y_outlook = dataset.iloc[:, -2].values

y_outlook = pd.DataFrame(y_outlook)

X_outlook_null = X_outlook[y_outlook[0].isnull()]
X_outlook_notnull = X_outlook[y_outlook[0].notnull()]

y_outlook_null = y_outlook[y_outlook[0].isnull()]
y_outlook_notnull = y_outlook[y_outlook[0].notnull()]

# plotting boxplot for outlook
box_plot = sns.boxplot(x=y_outlook_notnull[0],y=X_outlook_notnull[0] )
box_plot.set(xlabel='Outlook', ylabel='High Temperature')
plt.show()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_outlook_notnull, y_outlook_notnull)

# Predicting the Test set results
y_outlook_notnull_pred = RF_classifier.predict(X_outlook_notnull)
y_outlook_notnull_pred  = pd.DataFrame(y_outlook_notnull_pred )

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_outlook_notnull, y_outlook_notnull_pred)
cr = classification_report(y_outlook_notnull, y_outlook_notnull_pred)

#Outlook NaN to Missisng
dataset['Outlook'] = dataset['Outlook'].fillna('Missing')

#Weekday is incorrect
dataset['Weekday'] = pd.to_datetime(dataset['Date']).dt.weekday_name

#unique code and store name
print(dataset.groupby(['Code', 'Store Name']).size().reset_index().rename(columns={0:'count'}))

#change code 20002 to 20488
dataset.Code.replace([20002], [20488], inplace=True)

#discount
dataset['Discount'] = dataset['Discount'].apply(lambda x: 0 if x <= 0 else 1)

#predicting no of Japnese Tourists
jt = dataset.loc[:,['Date','YenWonRatio','Holiday','ActualHighTemp','Outlook','Month','Weekday', 'Japanese Tourists']]

#distribute plot for Actual High Temp
sns.distplot(jt['ActualHighTemp'])
plt.show()

#encoding Actual high temp into catagory
jt['ActualHighTemp'] = jt['ActualHighTemp'].apply(lambda x: 0 if x <= 60 else 1)

#distribute plot for Actual High Temp
sns.distplot(jt['ActualHighTemp'])
plt.show()

#encode Outlook to catagories
labelencoder_outlook = LabelEncoder()
jt.iloc[:, 4] = labelencoder_outlook.fit_transform(jt.iloc[:, 4])
labelencoder_outlook.classes_

#encode Weekday to catagories
labelencoder_weekday = LabelEncoder()
jt.iloc[:, 6] = labelencoder_weekday.fit_transform(jt.iloc[:, 6])
labelencoder_weekday.classes_

#dummies for Outlook
encode_outlook = pd.get_dummies(jt['Outlook'],  prefix_sep='_', drop_first=True)

jt = pd.concat([jt,encode_outlook], axis = 1)

jt.drop(['Outlook'], axis = 1, inplace = True)

#dummies for Month
encode_month = pd.get_dummies(jt['Month'],  prefix_sep='_', drop_first=True)

jt = pd.concat([jt,encode_month], axis = 1)

jt.drop(['Month'], axis = 1, inplace = True)

#dummies for weekday
encode_weekday = pd.get_dummies(jt['Weekday'],  prefix_sep='_', drop_first=True)

jt = pd.concat([jt,encode_weekday], axis = 1)

jt.drop(['Weekday'], axis = 1, inplace = True)

#get only not null data
X_jt_notnull = jt[jt['Japanese Tourists'].notnull()]

#drop duplicate data
X_jt_notnull = X_jt_notnull.drop_duplicates()
y_jt_notnull = X_jt_notnull.iloc[:,4].values
X_jt_notnull.drop(['Japanese Tourists'], axis = 1, inplace = True)
X_jt_notnull.drop(['Date'], axis = 1, inplace = True)

# get only null data
X_jt_null = jt[jt['Japanese Tourists'].isnull()]
X_jt_null = X_jt_null.drop_duplicates()
X_jt_null.drop(['Japanese Tourists'], axis = 1, inplace = True)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_jt_notnull, y_jt_notnull, test_size = 1/3, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
Lin_reg_jt = LinearRegression()
Lin_reg_jt.fit(X_train, y_train)

#prediction for training set
y_jt_train_pred = Lin_reg_jt.predict(X_train)

#Plotting predictions
plt.plot(range(1,209), y_train, color='g',  label='Actual')
plt.plot(range(1,209), y_jt_train_pred, color='b',  label='Predicted')
plt.ylabel('Num of Japanese Tourists for train dataset')
plt.legend()
plt.show()

from sklearn import metrics
print(metrics.mean_squared_error(y_train,y_jt_train_pred))
print(metrics.mean_absolute_error(y_train,y_jt_train_pred))

#prediction for test set
y_jt_test_pred = Lin_reg_jt.predict(X_test)

#Plotting predictions
plt.plot(range(1,105), y_test, color='g',  label='Actual')
plt.plot(range(1,105), y_jt_test_pred, color='b',  label='Predicted')
plt.ylabel('Num of Japanese Tourists for test dataset')
plt.legend()
plt.show()


print(metrics.mean_squared_error(y_test,y_jt_test_pred))
print(metrics.mean_absolute_error(y_test,y_jt_test_pred))

#OLS model
import statsmodels.formula.api as sm
X_opt = np.append(arr = np.ones((208, 1)).astype(int), values = X_train, axis = 1)

regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()

regressor_OLS.summary()

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = np.append(arr = np.ones((208, 1)).astype(int), values = X_train, axis = 1)
X_Modeled = backwardElimination(X_opt, SL)
X_Modeled =pd.DataFrame(X_Modeled)
X_Modeled=X_Modeled.iloc[:,1:].values
X_Modeled =pd.DataFrame(X_Modeled)

regressor_OLS = sm.OLS(endog = y_train, exog = X_Modeled).fit()
regressor_OLS.summary()

#Training the train set with optimized training set
Lin_reg_jt_OLS = LinearRegression()
Lin_reg_jt_OLS.fit(X_Modeled, y_train)

#predicting for training set
y_jt_train_pred = Lin_reg_jt_OLS.predict(X_Modeled)

#Plotting predictions
plt.plot(range(1,209), y_train, color='g',  label='Actual')
plt.plot(range(1,209), y_jt_train_pred, color='b',  label='Predicted')
plt.ylabel('Num of Japanese Tourists for train dataset for optimized model')
plt.legend()
plt.show()


print(metrics.mean_squared_error(y_train,y_jt_train_pred))
print(metrics.mean_absolute_error(y_train,y_jt_train_pred))


#prediction for test set
y_jt_test_pred = Lin_reg_jt.predict(X_test)

#Plotting predictions
plt.plot(range(1,105), y_test, color='g',  label='Actual')
plt.plot(range(1,105), y_jt_test_pred, color='b',  label='Predicted')
plt.ylabel('Num of Japanese Tourists for test dataset')
plt.legend()
plt.show()


print(metrics.mean_squared_error(y_test,y_jt_test_pred))
print(metrics.mean_absolute_error(y_test,y_jt_test_pred))


#Get prediction of missing no jap tour
y_null = Lin_reg_jt.predict(X_jt_null.iloc[:,1:].values)

JTP = pd.concat([pd.DataFrame(X_jt_null.iloc[:,0].values),pd.DataFrame(y_null)], axis = 1)

no_of_JT = JTP
no_of_JT.columns = ['Date','Num of JT']

dataset.merge(no_of_JT, on='Date', )

dataset2 = pd.merge(dataset, no_of_JT, on='Date', how = 'left')

dataset2['Japanese Tourists']= dataset2['Japanese Tourists'].combine_first(dataset2['Num of JT'])

dataset3 = dataset2.loc[:,['Code', 'Discount', 'Japanese Tourists', 'Total Sales']]

encode_sales = pd.get_dummies(dataset3['Code'],  prefix_sep='_', drop_first=True)

dataset3 = pd.concat([dataset3,encode_sales], axis = 1)

dataset3.drop(['Code'], axis = 1, inplace = True)

y = dataset3.iloc[:,2].values
dataset3.drop(['Total Sales'], axis = 1, inplace = True)
X = dataset3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Multiple Linear Regression to the Training set

Lin_reg = LinearRegression()
Lin_reg.fit(X_train, y_train)

y_pred_l = Lin_reg.predict(X_train)
y_train = pd.DataFrame(y_train)

#Plotting predictions
plt.plot(range(1,1699), y_train, color='g', label='Actual')
plt.plot(range(1,1699), y_pred_l, color='b',label='Predicted')
plt.ylabel('Total sales for training dataset LR')
plt.legend()
plt.show()


print(metrics.mean_squared_error(y_train,y_pred_l))
print(metrics.mean_absolute_error(y_train,y_pred_l))


X_opt = np.append(arr = np.ones((1698, 1)).astype(int), values = X_train, axis = 1)

SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)
X_Modeled =pd.DataFrame(X_Modeled)
X_Modeled=X_Modeled.iloc[:,1:].values

X.drop(['Discount'], axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_train, y_train)
y_pred_l = lin_reg_2.predict(X_train)

#Plotting predictions
plt.plot(range(1,1699), y_train, color='g', label = 'Actual')
plt.plot(range(1,1699), y_pred_l, color='b', label = 'Predicted')
plt.ylabel('Total sales for training dataset LR optimized')
plt.legend()
plt.show()


# Fitting second degree Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 1/3, random_state = 0)
lin_reg_2.fit(X_train, y_train)
y_pred_l = lin_reg_2.predict(X_train)

#Plotting predictions
plt.plot(range(1,1699), y_train, color='g', label = 'Actual')
plt.plot(range(1,1699), y_pred_l, color='b', label = 'Predicted')
plt.ylabel('Total sales for training dataset LR optimized second degree polynomial')
plt.legend()
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 1/3, random_state = 0)
lin_reg_2.fit(X_train, y_train)
y_pred_l = lin_reg_2.predict(X_train)

#Plotting predictions
plt.plot(range(1,1699), y_train, color='g', label = 'Actual')
plt.plot(range(1,1699), y_pred_l, color='b', label = 'Predicted')
plt.ylabel('Total sales for training dataset LR optimized third degree polynomial')
plt.legend()
plt.show()


y_pred_test = lin_reg_2.predict(X_test)

#Plotting predictions
plt.plot(range(1,850), y_test, color='g', label = 'Actual')
plt.plot(range(1,850), y_pred_test, color='b', label = 'Predicted')
plt.ylabel('Total sales for Test dataset LR optimized third degree polynomial')
plt.legend()
plt.show()

box_plot = sns.boxplot(x=dataset['Code'],y=dataset['Total Sales'])

box_plot = sns.boxplot(x=dataset['Distance from Main Street(Feet)'],y=dataset['Total Sales'] )
plt.show()
box_plot = sns.boxplot(x=dataset['Distance from Station X(Feet)'],y=dataset['Total Sales'] )
plt.show()
box_plot = sns.boxplot(x=dataset['Distance from Station Y(Feet)'],y=dataset['Total Sales'] )
plt.show()

box_plot = sns.boxplot(x=dataset2['Month'],y=dataset2['Japanese Tourists'] )
plt.show()
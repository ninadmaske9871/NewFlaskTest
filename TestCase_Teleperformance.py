# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 19:12:25 2022

@author: ninad_000
"""

"""  

  *********** Importing the libraries **********  

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc


import pickle

""" 
*******************************************************************
***********  Importing the Dataset Training and Testing *********** 
*******************************************************************
"""



missing_values = ["n/a", "na", "--","?","NA","NaN"]
df=pd.read_csv(r"TrainingData_V1.csv", na_values = missing_values)



#df_test=pd.read_csv(r"C:\python\TestingData_For_Candidate.csv", na_values = missing_values)

df_test=pd.read_excel(r"TestingData_For_Candidate.xlsx", na_values = missing_values)



""" Removing Leading and trailing spaces """


df.columns=df.columns.str.strip()
df_test.columns=df_test.columns.str.strip()


""" 
**************************************************
***********  Exploratory Data Analysis *********** 
**************************************************
"""

print(df.shape)
print(df.head())
print(df.tail())
print(df.describe())

print(df.info())
print(df_test.info())


# Missing Values

print(df.isnull().sum())

print("Total missing values ---> ", df.isnull().sum().sum())


# Visual Representation of missing values
msno.bar(df)
msno.matrix(df)
msno.heatmap(df)

print(df.nunique())

# dropping order_item__id as this is just an identifier for an order.

# user_title  Values
print("\n *** item_size  Values ***")
print(df["item_size"].value_counts())
# item_color  Values
print("\n *** item_color  Values ***")
print(df["item_color"].value_counts())
# brand_id  Values
print("\n *** brand_id  Values ***")
print(df["brand_id"].value_counts())
# user_title  Values
print("\n *** user_title  Values ***")
print(df["user_title"].value_counts())





# to find which user_title group returns the most
print(df[df["return"]==1].groupby(by="user_title")["return"].count().sort_values(ascending=False))

# to find which item is being returned the most
print(df[df["return"]==1].groupby(by="item_id")["return"].count().sort_values(ascending=False))

# to find which item size is being returned the most
print(df[df["return"]==1].groupby(by="item_size")["return"].count().sort_values(ascending=False))

# to find which item size is being returned the most
rtemp = (df[df["return"]==1].groupby(by="item_color")["return"].count().sort_values(ascending=False))



# print(rtemp.to_string())

""" 
********************************************
***********  Feature Engineering *********** 
********************************************
"""

# deleting the irrelevant features

df = df.drop(["order_item_id"],axis=1)


# New User or Old User

df['user_reg_date'] = pd.to_datetime(df['user_reg_date'], format='%d-%m-%Y', errors='coerce') 
thisDay = datetime.datetime.today().strftime("%Y-%m-%d")
thisDay =  pd.to_datetime(thisDay,format='%Y-%m-%d')

df["New_Old_User"]=(thisDay-df["user_reg_date"])
df["New_Old_User"]=(thisDay-df["user_reg_date"]).dt.days/365.2425
df["New_Old_User"]=df["New_Old_User"].fillna(df["New_Old_User"].mean())



#df = df.drop(["user_reg_date"],axis=1)


# Calculate the Users Age for consideration

df['user_dob'] = pd.to_datetime(df['user_dob'], format='%d-%m-%Y', errors='coerce') 
thisDay = datetime.datetime.today().strftime("%Y-%m-%d")
thisDay =  pd.to_datetime(thisDay,format='%Y-%m-%d')

df["Age"]=(thisDay-df["user_dob"])
df["Age"]=(thisDay-df["user_dob"]).dt.days/365.2425
df["Age"]=df["Age"].fillna(df["Age"].mean())

df = df.drop(["user_dob"],axis=1)


# Calculate the order month for seasonality

df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y', errors='coerce')

df['order_month'] = df['order_date'].apply(lambda x: x.strftime('%m'))
df['order_month'] = df['order_month'].astype(int,errors='ignore')

print(df.nunique())
# to find which user_title group returns the most
print(df[df["return"]==1].groupby(by="order_month")["return"].count().sort_values(ascending=False))



# Calculate the number of days to deliver

df['delivery_date'] = pd.to_datetime(df['delivery_date'], format='%d-%m-%Y', errors='coerce') 
df['delivery_time']=(df['delivery_date']-df["order_date"]).dt.days
df['delivery_time']=df['delivery_time'].fillna(df['delivery_time'].mean())
df = df.drop(['order_date','delivery_date'],axis=1)

df['delivery_time'] = df['delivery_time'].clip(lower=0)
print(df.nunique())
print(df.isnull().sum())


# Calculate price product cheap or expensive

df_sub = df[['item_id', 'item_price']]
df_sub['avg_item_price'] = df_sub.groupby(by=['item_id']).transform('mean')
df['price_expensive_cheap'] = df['item_price'] - df_sub['avg_item_price']



# Data Imputation for missing values in Item Color

df["item_color"].fillna(df["item_color"].mode()[0], inplace=True)
print(df.isnull().sum())


# Data standardisation for Item Size


def itemSizeConverter(x):
    x=x.replace("+","")
    if x.isnumeric()==False:  
        if x=='xxxl':
            x=48
        elif x=='xxl':
            x=44
        elif x=='xl':
            x=42
        elif x=='l':
            x=40
        elif x=='m':
            x=38
        elif x=='s':
            x=36
        elif x=='xs':
            x=34
        else: 
            x=40
    return int(x)

df['item_size_standardised'] = df.item_size.apply(itemSizeConverter)

df = df.drop(["item_size","user_reg_date"],axis=1)

print(df.nunique())


print(df.info())

#rtemp = (df[df["return"]==1].groupby(by="item_size_standardised")["return"].count().sort_values(ascending=False))

# rtemp = (df[df["return"]==0].groupby(by="user_id")["return"].count().sort_values(ascending=False))

# file = open('data.txt','w')
# file.write(rtemp.to_string())
# file.close()


# Converting user_title categorical values to numerical values

label_encoder = preprocessing.LabelEncoder()

print(df['user_title'].unique())
df['user_title']= label_encoder.fit_transform(df['user_title'])
print(df['user_title'].unique())
print(df.info())


# Converting item_color categorical values to numerical values

print(df['item_color'].unique())
df['item_color']= label_encoder.fit_transform(df['item_color'])
print(df['item_color'].unique())
print(df.info())



# Generating visualization for feature selection techniques.

# price_expensive_cheap
sns.set_theme(style="ticks", color_codes=True)
sns.catplot(x="return", y="price_expensive_cheap", data=df)

df["Temp"] = df['price_expensive_cheap'] < 0

print(df[df["return"]==1].groupby(by="Temp")["return"].count().sort_values(ascending=False))
dfnew = df.groupby(by="Temp")
print(dfnew.size())
df = df.drop(["Temp"],axis=1)

# New_Old_User

sns.set_theme(style="ticks", color_codes=True)
sns.catplot(x="return", y="New_Old_User", data=df)

# Users_Age

sns.set_theme(style="ticks", color_codes=True)
sns.catplot(x="return", y="Age", data=df)

df['Age_Rounded']=df['Age'].round(decimals = 0)
dftemp = (df[df["return"]==1].groupby(by="Age_Rounded")["return"].count().sort_values(ascending=False))
df = df.drop(["Age_Rounded"],axis=1)


fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(x='return', y='Age', data=df, ax=ax)
ax.set_title('Returns based on Age')


print(dftemp.to_string())

#df.to_csv("C:\python\MyAnalysis.csv",index=False)

# order_month

sns.set_theme(style="ticks", color_codes=True)
sns.catplot(x="return", y="order_month", data=df)
sns.catplot(x="return", y="order_month", kind="bar", data=df)

# ‘hue’ is used to visualize the effect of an additional variable to the current distribution.
sns.countplot(df['return'], hue=df['order_month'])


# delivery_time

sns.catplot(x="order_month", y="return", kind="bar", data=df)

fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(x='return', y='delivery_time', data=df, ax=ax)
ax.set_title('Returns based on TimeToDeliver')

# g = sns.catplot(x="return", y="delivery_time", kind="violin", inner=None, data=df)
# sns.swarmplot(x="return", y="delivery_time", color="k", size=3, data=df, ax=g.ax)


fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(x='return', y='delivery_time', data=df, ax=ax)
ax.set_title('Returns based on TimeToDeliver')

#item_color

sns.countplot(df['return'], hue=df['item_color'])



print("done!!")

corr = df.corr()
plt.figure(figsize=(18,10))
sns.heatmap(corr,annot=True)


sns.countplot(x=df["return"]).set_title("Return Count")

sns.pairplot(data=df, x_vars=['return'],
                  y_vars=['item_color', 'brand_id', 'item_price','user_title','Age','order_month','delivery_time','item_size_standardised','New_Old_User','price_expensive_cheap'])


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True);
plt.title('Correlation Matrix', fontsize=20);


"""


**********************************
***********  Modelling *********** 
**********************************
"""

x = df.drop(['return'],axis=1) # training set
y = df['return'] # target set
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=8)

# scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

accuracy_list = []
f1_list = []
roc_auc_list = []



def result(model, modelName):
    
    #fit on data
    model.fit(X_train, y_train)
    
    print(model)
    
    #prediction
    pred = model.predict(X_test)
    
    #performance of model
    print("Classification Report: ", modelName ,"\n", classification_report(y_test, pred))
    print("-" * 100)
    print()
    
    #accuracy of model
    acc = accuracy_score(y_test, pred)
    accuracy_list.append([modelName,acc])
    print("Accuracy Score ->", modelName , acc)
    print("-" * 100)
    print()

    #f1-score of model
    f1 = f1_score(y_test, pred)
    f1_list.append([modelName,f1])
    print("F1 Score ->", modelName , f1)
    print("-" * 100)
    print()

    #roc-auc curve of model
    fpr,tpr,threshold = roc_curve(y_test,pred)
    auc_value = auc(fpr,tpr)
    rocauc_score = roc_auc_score(y_test, pred)
    roc_auc_list.append([modelName,rocauc_score])
    plt.figure(figsize=(5,5),dpi=100)
    print("ROC-AUC Score -> ", modelName , f1)
    print("-" * 100)
    print()
    plt.plot(fpr,tpr,linestyle='-',label = "(auc_value = %0.3f)" % auc_value)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    print()
 
   #confusion matrix for model
    tag = " : Confusion Matrix"
    modelName = modelName + tag
    plt.figure(figsize=(10, 5))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g');
    plt.title(modelName, fontsize=20)




# Logistic Regression

lr = LogisticRegression()
result(lr,'LogisticRegression')



# Decision Tree

lr = DecisionTreeClassifier()
result(lr,'DecisionTree')



#K-Nearest Neighbors
lr = KNeighborsClassifier()
result(lr,'KNN')


# Random Forest 
lr = RandomForestClassifier()
result(lr,'RandomForest')


# XGBoost
lr = XGBClassifier()
result(lr,'XGBoost')



print("Accuracy Score -> ", accuracy_list)
print("F1 Score -> ", f1_list)
print("ROC-AUC Score -> ", roc_auc_list)



# Random Forest and XGBoost give accuracy of 65 % 

# HyperParameter Tuning



from sklearn.model_selection import RandomizedSearchCV
parameters = {
    "n_estimators": [50,100,150,200,300], 
    "min_samples_leaf":[1,3,5,7,9],
    "min_samples_split":[2,3,4,5,7,8],
    "max_features":[1,5,10,15,19]
}


random_forest = RandomForestClassifier()
model_random_forest = RandomizedSearchCV(
    estimator=random_forest, 
    param_distributions=parameters)


model_random_forest.fit(X_train, y_train)

model_random_forest.best_params_, model_random_forest.best_score_

finalModel_rf = RandomForestClassifier(**model_random_forest.best_params_)
finalModel_rf.fit(X_train, y_train)
y_preds = finalModel_rf.predict(X_test)

## Final f1 score & accuracy score
print(f1_score(y_preds, y_test))
print(accuracy_score(y_preds, y_test))





from sklearn.model_selection import RandomizedSearchCV

xgb = XGBClassifier()  ## Model to tune

paramSearchSpace = {
    'n_estimators' : [10, 50, 100,200],  ## Number of trees
    'gamma' : [1, 0.05, 0.1],    ## Regularisation parameter
    'scale_pos_weight' : [60, 70, 80] # Num pos / num Neg
}


model_xgb = RandomizedSearchCV(xgb, param_distributions=  paramSearchSpace)

# Fit with data
model_xgb.fit(X_train, y_train)
print(model_xgb.best_params_, model_xgb.best_score_)


finalModel = XGBClassifier(**model_xgb.best_params_)
finalModel.fit(X_train, y_train)
y_preds = finalModel.predict(X_test)

## Final f1 score & accuracy score
print(f1_score(y_preds, y_test))
print(accuracy_score(y_preds, y_test))





""" 
# ***********************************
# ***********  Prediction *********** 
# ***********************************
"""

### for testSet Prediction
df_test = df_test.drop(["order_item_id"],axis=1)

##########################

#df_test['user_reg_date'] = pd.to_datetime(df_test['user_reg_date'], format='%d-%m-%Y', errors='coerce') 
thisDay = datetime.datetime.today().strftime("%Y-%m-%d")
thisDay =  pd.to_datetime(thisDay,format='%Y-%m-%d')

df_test["New_Old_User"]=(thisDay-df_test["user_reg_date"])
df_test["New_Old_User"]=(thisDay-df_test["user_reg_date"]).dt.days/365.2425
df_test["New_Old_User"]=df_test["New_Old_User"].fillna(df_test["New_Old_User"].mean())

df_test = df_test.drop(["user_reg_date"],axis=1)

#########################
df_test['user_dob'] = pd.to_datetime(df_test['user_dob'], format='%d-%m-%Y', errors='coerce') 
thisDay = datetime.datetime.today().strftime("%Y-%m-%d")
thisDay =  pd.to_datetime(thisDay,format='%Y-%m-%d')

df_test["Age"]=(thisDay-df_test["user_dob"])
df_test["Age"]=(thisDay-df_test["user_dob"]).dt.days/365.2425
df_test["Age"]=df_test["Age"].fillna(df_test["Age"].mean())

df_test = df_test.drop(["user_dob"],axis=1)


#########################

df_test['order_date'] = pd.to_datetime(df_test['order_date'], format='%d-%m-%Y', errors='coerce')

df_test['order_month'] = df_test['order_date'].apply(lambda x: x.strftime('%m'))
df_test['order_month'] = df_test['order_month'].astype(int,errors='ignore')

#########################

df_test['delivery_date'] = pd.to_datetime(df_test['delivery_date'], format='%d-%m-%Y', errors='coerce') 
df_test['delivery_time']=(df_test['delivery_date']-df_test["order_date"]).dt.days
df_test['delivery_time']=df_test['delivery_time'].fillna(df_test['delivery_time'].mean())
df_test = df_test.drop(['order_date','delivery_date'],axis=1)

df_test['delivery_time'] = df_test['delivery_time'].clip(lower=0)

#########################


df_sub = df_test[['item_id', 'item_price']]
df_sub['avg_item_price'] = df_sub.groupby(by=['item_id']).transform('mean')
df_test['price_expensive_cheap'] = df_test['item_price'] - df_sub['avg_item_price']


#########################

print(df_test.info())

df_test["item_color"].fillna(df_test["item_color"].mode()[0], inplace=True)

def itemSizeConverter(x):
    
    x=str(x).replace("+","")
    type(x)
    if x.isnumeric()==False:  
        if x=='xxxl':
            x=48
        elif x=='xxl':
            x=44
        elif x=='xl':
            x=42
        elif x=='l':
            x=40
        elif x=='m':
            x=38
        elif x=='s':
            x=36
        elif x=='xs':
            x=34
        else: 
            x=40
    return int(x)

df_test['item_size_standardised'] = df_test.item_size.apply(itemSizeConverter)

df_test = df_test.drop(["item_size"],axis=1)

#########################

label_encoder = preprocessing.LabelEncoder()

print(df_test['user_title'].unique())
df_test['user_title']= label_encoder.fit_transform(df_test['user_title'])
print(df_test['user_title'].unique())
print(df_test.info())

#########################

print(df_test['item_color'].unique())
df_test['item_color']= label_encoder.fit_transform(df_test['item_color'])
print(df_test['item_color'].unique())
print(df_test.info())


norm = StandardScaler()
test_data_final_normalised = norm.fit_transform(df_test)

Final_Prediction = finalModel.predict(test_data_final_normalised) 

#Final DF to attach predicitons to
Final_DF = df_test.loc[:, ['item_size_standardised']] # target variable
Final_DF['Predictions']=Final_Prediction

Final_DF.to_csv("Result.csv",index=False)

pred_prob2 = finalModel.predict_proba(X_test)


""" 
# ***********************************
# ***********  Deployment *********** 
# ***********************************
"""

pickle.dump(finalModel, open('model.pkl','wb'))

#Welcome back! This file is an attempt to do some feature engineering and use Ridge, Lasso and ElasticNet on some actual data. I got the Algerian forest fires dataset from Kaggle. Here's the link {https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset?resource=download}. The code is similar to Krish Naik, but I had to figure out some things and change them while doing the data cleaning segment. Anyways, enjoy!


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import heatmap
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
%matplotlib inline



dataset = pd.read_csv('Algerian_forest_fires_dataset.csv')
dataset.head()
dataset.isnull().sum()

dataset.isnull().sum()

dataset[dataset.isnull().any(axis=1)]

#using drop here because there is only one row with 1 NA value in the data. 
dataset.dropna().reset_index(drop=True)
dataset.head()

dataset.iloc[[122]]
df=dataset.drop(122).reset_index(drop=True)
df.head()
df.iloc[[122]]
df=df.drop(122).reset_index(drop=True)
df.iloc[[122]]
df=df.drop(122).reset_index(drop=True)
df.iloc[[122]]


df.columns
df.info()

df[['day','month','year','Temperature',' RH',' Ws',]]=df[['day','month','year','Temperature',' RH',' Ws']].astype(int)
df[['Rain ','FFMC','DMC','DC','ISI','BUI','FWI']]=df[['Rain ','FFMC','DMC','DC','ISI','BUI','FWI']].astype(float)
df.info()

df=df.drop(['day','month','year'],axis=1)
df.head()

cleaned=df
cleaned.head()
cleaned['Classes  '] = np.where(df['Classes  '].str.contains('not', case=False, na=False), 0, 1)
print(cleaned.columns)
cleaned.dtypes
print(cleaned['Classes  '][120])

print(cleaned.columns)
cleaned.drop(columns='Region',inplace=True)
cleaned.head()
cleaned["Region"] = 0
cleaned.loc[124:, 'Region']=1

print(cleaned.head())
print(cleaned.tail())
sns.heatmap(cleaned.corr(),annot=True)
sns.heatmap(cleaned.corr(),annot=True)

cleaned.dtypes

cleaned.to_csv('Cleaned_fires.csv')


#Started the next part in a new file. so what you see as cleaned before is now df.

df= pd.read_csv('Cleaned_fires.csv')
df.head()
sns.heatmap(df.corr(), annot=True)

df['Classes  '].value_counts()

X=df.drop('FWI',axis=1)
y=df['FWI'] 

print(f'X is {X.head()}')
print(1)
print(f'Y is{y.head()}')


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

X_train.shape, X_test.shape

sns.heatmap(X_train.corr(), annot=True)


# finding columns with correlation more the a threshold value, set at 90% (from the course) This aims to reduce multicollinearity)

def correlation(dataset,threshold):
    correlated = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated.add(colname)
    return correlated

drop_features  = correlation(X_train,0.90)
print(drop_features)

X_train.drop(drop_features, axis = 1,inplace=True)
X_test.drop(drop_features, axis = 1,inplace=True)
X_test.head()
print(f'X_train is {X_train.head()}')

#feature Scaling/standardisation

scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled =scaler.transform(X_test)

X_test_scaled

plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title('Before')
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title('After')


linregress = LinearRegression()
linregress.fit(X_train_scaled,y_train)
y_pred = linregress.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print(f'MAE is {mae}')
print(f'R2 is {score}')
plt.scatter(y_test,y_pred)

lasso = Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print(f'MAE is {mae}')
print(f'R2 is {score}')
plt.scatter(y_test,y_pred)

lassocv=LassoCV(cv=5)
lassocv.fit(X_train_scaled,y_train)
lassocv.alphas_
y_pred = lassocv.predict(X_test_scaled)
plt.scatter(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print(f'MAE is {mae}')
print(f'R2 is {score}')

ridgecv=RidgeCV(cv=5)
ridgecv.fit(X_train_scaled,y_train)
y_pred=ridgecv.predict(X_test_scaled)
plt.scatter(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print(f'MAE is {mae}')
print(f'R2 is {score}')
print(f'R2 is {score}')


elasticnet= ElasticNet()
elasticnet.fit(X_train_scaled,y_train)
y_pred = elasticnet.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print(f'MAE is {mae}')
print(f'R2 is {score}')
plt.scatter(y_test,y_pred)


# Here is the assignment I did for EDA with the California housing dataset. It goes with the streamlit app I made in the cali_app.py file.

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline



# Tried to use the inbuilt data from sklearn but kept hitting some ssl erros. To get going with the work I downloaded the data from the internet. Once I have a stable internet connection I will try and use the pre-exisitng data.

# In[221]:


df=pd.read_csv('housing.csv')

#creating copies to make sure all different approaches used in the file run
df_knn = df
df_dummies = df


# In[222]:


df.head()


# In[223]:


print(df.dtypes)

df.describe()


# ### Summary Statistics and Outliers
# 
# I built the function because I didnt want to go over all the columns again and again.
# I also just used one visualisation because I don't know what else to plot.

# In[224]:


def plot_without_processing(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    df.hist(ax=ax, bins=30, edgecolor='black')
    plt.tight_layout()
    return fig

# In[229]:


def outliers(data):

    summary_data = {}
    for col in df.select_dtypes(include=['number']).columns:
        print(col)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

        summary_data[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers': outlier_count
        }

    return summary_data

summarised_data = outliers(df)
print(summarised_data)

summary_table = pd.DataFrame.from_dict(summarised_data, orient='index')
print(summary_table)

def remove_outliers(data):
    df=data.copy()
    for col in data.select_dtypes(include=['number']).columns:
        #print(col)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound), np.nan)

    return df
no_outliers = remove_outliers(df)
print('no outliers ')
print(no_outliers)

def use_knnimputer(df):
    knn= KNNImputer(n_neighbors=5)
    o_p = df['ocean_proximity']

    for_na_replace = df.drop('ocean_proximity', axis=1)


    for_na_replace= pd.DataFrame(knn.fit_transform(for_na_replace), columns = for_na_replace.columns)

    df = pd.concat([for_na_replace, o_p], axis=1)

    return df
#the remove outliers replaces outlier values with NaN. I did this because the number of outliers ended up making the model super inaccurate with the MAE going above 500.I used the KNN imputer to replace the nan values. I tried using the median first but it gave me a r2 score around 0.3-0.4. I wanted to see if I could improve that so i switched to the KNN method.

no_outliers = use_knnimputer(no_outliers)



# In[231]:


def plot_processed(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_scaled, ax=ax)
    plt.xticks(rotation=45)

    return fig

processed_boxplot = plot_processed(no_outliers)

# In[232]:

def no_o_hist(no_outliers):
    fig, ax = plt.subplots(figsize=(12, 6))
    no_outliers.hist(ax=ax, bins=30, edgecolor='black')
    plt.tight_layout()
    return fig



# In[233]:


unique_ocean = df['ocean_proximity'].unique()


# In[234]:


# Two approaches were available so I tried both: KNN Imputer from sklearn and Median Imputation.


# In[235]:

def do_encoding(no_outliers):

    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(no_outliers[['ocean_proximity']]).toarray()


    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ocean_proximity']))

    no_outliers= pd.concat([no_outliers, df_encoded], axis=1).drop(columns=['ocean_proximity'])

    return no_outliers

no_outliers = do_encoding(no_outliers)



# In[236]:




# In[237]:

def scaler_minmax(no_outliers):
    scaler2 = MinMaxScaler()

    df_scaled2 = pd.DataFrame(scaler2.fit_transform(no_outliers), columns=no_outliers.columns)
    return df_scaled2

df_scaled = scaler_minmax(no_outliers)
print(df_scaled)
df_scaled.isnull().sum()


# In[238]:


# I am using the heatmap to find the correlations and drop columns to avoid Multi collinearity. I'm trying to follow the process from the udemy course.
def corr_heat(df_scaled):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_scaled.corr(), annot=True)
    plt.tight_layout()
    return fig

heatmap = corr_heat(df_scaled)



# In[239]:


def find_corr(corr, threshold=0.85):
    drop_columns = []
    checked_pairs = set()
    for col in corr.columns:

        for row in corr.index:

            if col != row:
                pair = tuple(sorted((row, col)))

                if pair not in checked_pairs:
                    corr_value = corr[col][row]

                    if abs(corr_value) > threshold:
                        drop_columns.append((row, col, corr_value))
                        checked_pairs.add(pair)

    drop_columns = pd.DataFrame(drop_columns, columns=['F1','F2','Correlation strength'])

    return drop_columns

print(find_corr(df_scaled.corr(),0.9))

df_scaled.isnull().sum()


# In[240]:


cleaned_data = df_scaled.drop(columns=['total_bedrooms','population'], axis=1)


# In[241]:


print(cleaned_data)

cleaned_data.isnull().sum()


# I'll be using median_house_value as the target for the regression.

# In[242]:

def split_data(cleaned_data):
    y=cleaned_data['median_house_value']
    X = cleaned_data.drop(columns=['median_house_value'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(cleaned_data)

# In[243]:

def linear_regression(X_train, X_test, y_train, y_test):
    linregress = LinearRegression()
    linregress.fit(X_train,y_train)
    y_pred = linregress.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return y_pred, mae, r2

y_pred, mae, r2 = linear_regression(X_train, X_test, y_train, y_test)


def show_linear_regression(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes
    ax.scatter(y_test, y_pred, label="Predicted vs Actual", alpha=0.7)
    x_vals = np.linspace(min(y_test), max(y_test), 100)
    ax.plot(x_vals, x_vals, color='red', linestyle='--', label="Perfect Fit (y=x)")
    return fig
# In the models ahead i have used the CV versions because I didnt know what alpha to choose.

# In[244]:

def run_ridgeCV(X_train, X_test, y_train, y_test):
    ridge = RidgeCV()
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, r2
y_pred,mae,r2 = run_ridgeCV(X_train, X_test, y_train, y_test)

def show_ridgeCV(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, label="Predicted vs Actual", alpha=0.7)
    x_vals = np.linspace(min(y_test), max(y_test), 100)
    ax.plot(x_vals, x_vals, color='red', linestyle='--')
    return fig

ridge_fig = run_ridgeCV(X_train, X_test, y_train, y_test)




# In[245]:

def run_lassoCV(X_train, X_test, y_train, y_test):
    lasso = LassoCV()
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, r2

y_pred,mae,r2 = run_lassoCV(X_train, X_test, y_train, y_test)

def show_lassoCV(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, label="Predicted vs Actual", alpha=0.7)
    x_vals = np.linspace(min(y_test), max(y_test), 100)
    ax.plot(x_vals, x_vals, color='red', linestyle='--')
    return fig

lasso_fig = show_lassoCV(y_test, y_pred)



# In[246]:

def run_enet(X_train, X_test, y_train, y_test):
    enet = ElasticNetCV()
    enet.fit(X_train, y_train)
    y_pred = enet.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, r2

y_pred,mae,r2 = run_ridgeCV(X_train, X_test, y_train, y_test)

def show_enet(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, label="Predicted vs Actual", alpha=0.7)
    x_vals = np.linspace(min(y_test), max(y_test), 100)
    ax.plot(x_vals, x_vals, color='red', linestyle='--')
    return fig
enet_fig = show_enet(y_test, y_pred)

# In[250]:

def use_gsCV(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    param_grid = [
        {'model': [Lasso()], 'model__alpha': [0.001, 0.01, 0.1, 1, 10]},
        {'model': [Ridge()], 'model__alpha': [0.001, 0.01, 0.1, 1, 10]},
        {'model': [ElasticNet()], 'model__alpha': [0.001, 0.01, 0.1, 1, 10], 'model__l1_ratio': [0.1, 0.5, 0.9]}
    ]

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params


best_model, best_params = use_gsCV(X_train, X_test, y_train, y_test)
print("Best Model:", best_model)
print("Best Parameters:",best_params)


#y_pred = grid_search.best_estimator_.predict(X_test)


#This is the streamlit app I build connected to the Cali_housing_assignment.py in this code when I import from EDA_assignment Im refering to the Cali_housing_assignment.py file.  
import streamlit as st
import pandas as pd

st.title("California housing data assignment")

df = pd.read_csv('housing.csv')

st.write('Here is a quick look at the California housing data ->')
st.write(df.head())

st.write("Let's try and get a sense of the data!")
st.write(df.describe())

st.write("Let's visualize the data.")
from EDA_assignment import plot_without_processing, summarised_data

st.pyplot(plot_without_processing(df))

from EDA_assignment import outliers
summarised_data = outliers(df)
st.write("Let's get a sense of the outliers!")
st.write(pd.DataFrame.from_dict(summarised_data, orient='index'))

from EDA_assignment import remove_outliers, use_knnimputer
no_outliers = use_knnimputer(remove_outliers(df))
st.write('I have done some outlier handling. Take a look...')

revised_data =  outliers(no_outliers)
st.write(pd.DataFrame.from_dict(revised_data, orient='index'))

from EDA_assignment import plot_processed
processed_plot = plot_processed(no_outliers)
st.write('Let me visualise the data after some outliers handling.')
st.pyplot(processed_plot)

from EDA_assignment import no_o_hist
no_o_hist = no_o_hist(df)
st.pyplot(no_o_hist)

from EDA_assignment import do_encoding
no_outliers = do_encoding(no_outliers)
st.write('I have encoded the Ocean Proximity data into numeric values using the OneHotEncoder')

from EDA_assignment import scaler_minmax
df_scaled = scaler_minmax(no_outliers)

from EDA_assignment import corr_heat
heatmap  = corr_heat(df_scaled)
st.write("I have also done some scaling. Now let's look at the correlations in the data.")
st.pyplot(heatmap)

from EDA_assignment import find_corr
drop_columns = find_corr(df_scaled.corr())
st.write("Cleaning that up, we can see with a threshold of 0.9, the correlations are")
st.write(drop_columns)
st.write("Latitude and longitude are now coincidentally correlated. Regarding the others, I will be removing Population and total_bedrooms.")

cleaned_data = df_scaled.drop(columns=['total_bedrooms','population'], axis=1)

from EDA_assignment import split_data
X_train, X_test, y_train, y_test = split_data(cleaned_data)

st.write("Now let's run some models")

from EDA_assignment import linear_regression, show_linear_regression
y_pred, mae, r2 = linear_regression(X_train, X_test, y_train, y_test)
chart_lin_reg = show_linear_regression(y_test, y_pred)
st.write("The results of linear regression are:")
st.pyplot(chart_lin_reg)
st.write("MAE=", mae)
st.write("R2=", r2)

from EDA_assignment import run_ridgeCV, show_ridgeCV
y_pred, mae, r2 = run_ridgeCV(X_train, X_test, y_train, y_test)
chart_ridge = show_ridgeCV(y_test, y_pred)
st.write("The results of ridge regression are:")
st.pyplot(chart_ridge)
st.write("MAE=", mae)
st.write("R2=", r2)

from EDA_assignment import run_lassoCV, show_lassoCV
y_pred, mae, r2 = run_lassoCV(X_train, X_test, y_train, y_test)
chart_lasso = show_lassoCV(y_test, y_pred)
st.write("The results of lasso regression are:")
st.pyplot(chart_lasso)
st.write("MAE=", mae)
st.write("R2=", r2)

from EDA_assignment import run_enet, show_enet
y_pred, mae, r2 = run_enet(X_train, X_test, y_train, y_test)
chart_enet = show_enet(y_test, y_pred)
st.write("The results of enet regression are:")
st.pyplot(chart_enet)
st.write("MAE=", mae)
st.write("R2=", r2)

st.write("Finally using the GridSearchCV module")
from EDA_assignment import use_gsCV
best_model, best_params = use_gsCV(X_train, X_test, y_train, y_test)
st.write("The results of grid search are:")
st.write("Best model",best_model)
st.write("Best parameters",best_params)

''' This file is related to all the work I've done related to the Feature engineering segment of the Krish Naik Udemy course. Im going to attempt to look at different cases related to the following topics:
1. Handling missing values: 
  - Mean, median and mode value imputation
2. Handling imbalanced datasets:
  - Up/down sampling
  - Small Minority Oversampling Technique(SMOTE)
3. Data Encoding
  - Nominal/OHE, Label, Ordinal'''


## Code starts:

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import numpy as np

# Original DataFrame
marbles = pd.DataFrame({
    'colour': ['red', 'blue', 'green', 'orange', 'yellow', 'purple'],
    'Number': [1, 2, 3, 4, 5, 6]
})


encoder = OneHotEncoder()
encoded = encoder.fit_transform(marbles[['colour']]).toarray()
encoded_df = pd.DataFrame(encoded)  # Encoded DataFrame without column names


marbles_2 = pd.concat([marbles, encoded_df], axis=1)
print("Marbles with Encoded Values:")
print(marbles_2)
print(2)

# after encoding it changes 'red' to 'colour_red'. I dont like it, so ive included the split part
def encoding_to_dict(input_df, encoder):
    color_names = encoder.get_feature_names_out()  # Get proper feature names
    new = {color.split('_')[1]: input_df.iloc[:, idx].tolist() 
           for idx, color in enumerate(color_names)}
    return new

encoded_dict = encoding_to_dict(encoded_df, encoder)
print("Encoded Dictionary:")
print(encoded_dict)
print(3)

encoded_df_final = pd.DataFrame(list(encoded_dict.items()), columns=['Colour', 'Encoded Values'])
print("Final Encoded DataFrame:")
print(encoded_df_final)



#code below is not finished.
tips= sns.load_dataset('tips')
print(tips)
encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
catagorical_variables=['sex','smoker','day','time']
encoded_variables=encoder.fit_transform(tips[catagorical_variables])
encoded_df = pd.DataFrame(encoded_variables, columns=encoder.get_feature_names_out(catagorical_variables))

# Concatenate the encoded DataFrame with the original DataFrame
final_df = pd.concat([tips, encoded_df], axis=1)
print(final_df)

days=tips['day'].unique()
days = days.tolist()
print(days)
days_df = pd.DataFrame({'Days': days})
print(days_df)
days_encoded = encoder.fit_transform(days_df[['Days']])
print(days_encoded)
days_df['encoding_key'] = np.nan  # Initialize empty column
print(days_df)

for idx, arr in enumerate(days_encoded):
    days_df.at[idx, 'encoding_key'] = [arr]  # Assign encoded array as list
print(days_df)

#days_encoded=encoder.fit_transform(days[['days']]).toarray()




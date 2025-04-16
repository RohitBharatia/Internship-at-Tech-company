## Assignment to make a sentiment analysis dashboard using the data sentiment file.
## Use streamlit for making the dashboard
## Recreate or replace
##  Boto3 and .bedrock is not needed. I will focus on first making a simple dashboard, and then integrate the rest later.
# Import
#%%

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import regex as re
import numpy as np
from io import BytesIO
from transformers import AutoTokenizer
import nltk
#nltk.download('vader_lexicon')
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#%%
# Functions for analysis
tokeniser = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenise_text(text):
    if not isinstance(text, str):
        return []  # or return np.nan if you want to skip
    tokens = tokeniser.tokenize(text)
    tokens = [t for t in tokens if re.match(r'^\w+$', t)]
    return tokens

def vader_sentiment_summary(df):
    analyzer = SentimentIntensityAnalyzer()
    df["compound_score"] = df['Review Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df["sentiment"] = np.where(df['compound_score'] >= 0.05, 'Positive', np.where(df['compound_score'] <= -0.05, 'Negative','Neutral'))
    return df['sentiment'].value_counts()





#%%
# Analysis


data = pd.read_csv('data_sentiment 1 1.csv')
data.head()
data.rename(columns={'Review Submit Date and Time':'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Month'] = pd.to_datetime(data['Date']).dt.month
years = data['Year'].unique()

#%%
def run_sentiment(year_in,data):
    selected_data = data[data['Year'] == year_in]
    selected_data = selected_data[['Review Text','Star Rating']]
    selected_data['Review Text'] = selected_data['Review Text'].apply(tokenise_text)
    selected_data['Review Text'] = selected_data['Review Text'].apply(lambda tokens: " ".join(tokens))
    scores = vader_sentiment_summary(selected_data)
    return scores


#%%
#



st.set_page_config("Dashboard", layout="wide")
st.title("Insights")

col1, col2,col3 = st.columns([3,3,3])
with col1:
    year_in = st.selectbox('Select a year',years,placeholder="Select a year" ,index=None)
    if year_in:
        scores = run_sentiment(year_in,data)

        def plot_sentiment(scores):
            fig,ax = plt.subplots()
            scores.plot(kind='bar', color=['green', 'grey', 'red'], ax=ax)
            ax.set_title('Sentiment Distribution')
            ax.set_ylabel('Count')
            ax.set_xlabel('Sentiment')
            return fig


        fig = plot_sentiment(scores)
        st.pyplot(fig)


with col2:

    selected_data = data[data['Year'] == year_in][['Review Text', 'Star Rating']]
    selected_year_data = data[data['Year'] == year_in]['Review Text'].astype(str)
    selected_month_data = data[data['Year'] == year_in]['Month'].astype(str)
    st.subheader(f"Star Rating Distribution for Year {year_in}")
    star_rating_counts = selected_data['Star Rating'].value_counts()
    labels = star_rating_counts.index
    sizes = star_rating_counts.values
    colors = ['#37bd79', '#a7e237', '#308fac', '#f4e604', '#0457ac']
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.4))
    plt.axis('equal')
    plt.legend(loc='upper right')
    st.pyplot(plt)
    # wordcloud
    # st.subheader("Word Clouds")

with col3:
    all_text = ' '.join(selected_year_data.dropna().tolist()).strip()

    if len(all_text.split()) > 0:
        wordcloud = WordCloud(width=800, height=550, background_color='white').generate(all_text)
        st.subheader("Word Cloud: What People Say")
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title("Word Cloud: What People Say")
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.warning("Not enough valid review text to generate a word cloud for this year.")

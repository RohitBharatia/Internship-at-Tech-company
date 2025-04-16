import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import regex as re
import numpy as np
from io import BytesIO
from transformers import AutoTokenizer
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#%%
data = pd.read_csv('/Users/rohit/Python_trials/streamlit_sentiment/data_sentiment 1 1.csv')
st.title('Sentiment Analysis - Raw Data')
data.rename(columns={'Review Submit Date and Time':'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Month'] = pd.to_datetime(data['Date']).dt.month
years = data['Year'].unique()
#%%
tokeniser = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenise_text(text):
    if not isinstance(text, str):
        return []
    tokens = tokeniser.tokenize(text)
    tokens = [t for t in tokens if re.match(r'^\w+$', t)]
    return tokens

def vader_sentiment_summary(df):
    analyzer = SentimentIntensityAnalyzer()
    df["compound_score"] = df['Review Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df["sentiment"] = np.where(df['compound_score'] >= 0.05, 'Positive', np.where(df['compound_score'] <= -0.05, 'Negative','Neutral'))
    return df['sentiment'].value_counts()


def run_sentiment(year_in,data):
    selected_data = data[data['Year'] == year_in]
    selected_data = selected_data[['Review Text','Star Rating']]
    selected_data['Review Text'] = selected_data['Review Text'].apply(tokenise_text)
    selected_data['Review Text'] = selected_data['Review Text'].apply(lambda tokens: " ".join(tokens))
    scores = vader_sentiment_summary(selected_data)
    return scores

def entries_per_year(data):
    return data['Year'].value_counts().sort_index().rename("Reviews").to_frame()


def get_rank():
    final_scores = {}

    for year in years:
        scores = run_sentiment(year,data)
        final_scores[year] = scores
    avg_sent_score = {}
    for key in final_scores.keys():
        avg_sent_score[key] = final_scores[key].mean()

    avg_sent_score = pd.DataFrame.from_dict(avg_sent_score, orient='index', columns=['Average Score'])
    return avg_sent_score

def show_rank(mean_score):
    fig, ax = plt.subplots()
    mean_score.plot(kind='bar', color=['green', 'grey', 'red'], ax=ax)
    ax.set_title('Sentiment Analysis Ranking')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Sentiment Score')
    return fig





#%%


data = data.drop(columns=['Unnamed: 0'])

# Initialize toggle state
if "show_head" not in st.session_state:
    st.session_state.show_head = False

# Toggle button
if st.button("Show/Hide Sample of Data"):
    st.session_state.show_head = not st.session_state.show_head

# Display data.head() if toggle is on
if st.session_state.show_head:
    st.dataframe(data.head())


if 'show_stats' not in st.session_state:
    st.session_state.show_stats = False

if st.button("Show/Hide Stats"):
    st.session_state.show_stats = not st.session_state.show_stats

if st.session_state.show_stats:
    plot = show_rank(get_rank())
    st.pyplot(plot)
    st.write('Total Entries')
    st.dataframe(entries_per_year(data))



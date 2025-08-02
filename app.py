import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Sentiment Analyzer Class
class SentimentAnalyzer:
    def __init__(self, model_name):
        self.llm = Ollama(model=model_name)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a sentiment analysis expert. Analyze the sentiment of the given text and respond with ONLY ONE WORD: 'positive', 'negative', or 'neutral'."),
            ("human", "Analyze the sentiment of the following text: {text}")
        ])
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def analyze_sentiment(self, text):
        result = self.chain.invoke({"text": text})
        sentiment = result.strip().lower()
        return {'classification': sentiment, 'confidence': 1.0}

# AI Chat Assistant Function
def chat_about_sentiment(user_question, analyzed_data):
    llm = Ollama(model='TinyLlama')
    prompt_text = f"""
You are an AI assistant that summarizes and answers questions about sentiment analysis results.
Given the following data:
{analyzed_data}

User Question: {user_question}

Provide a short and clear answer.
"""
    result = llm.invoke(prompt_text)
    return result.strip()

@st.cache_resource
def get_sentiment_analyzer(model_name):
    return SentimentAnalyzer(model_name)

def create_word_cloud(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word.lower() for word in word_tokens if word.isalnum() and word.lower() not in stop_words]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_text))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_sentiment_gauge(confidence, sentiment):
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment: {sentiment.capitalize()}"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': colors.get(sentiment, 'gray')},
            'steps': [
                {'range': [0, 0.33], 'color': "lightgray"},
                {'range': [0.33, 0.66], 'color': "gray"},
                {'range': [0.66, 1], 'color': "darkgray"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': confidence}
        }
    ))
    return fig

# Main Dashboard Layout
def main():
    st.set_page_config(page_title="Sentiment Analysis Dashboard with AI Chatbot ", page_icon="👁️🔍", layout="wide")
     # Custom CSS for background and styling
    st.markdown("""
        <style>
            .main {
                background: linear-gradient(to right, #f8f9fa, #e9ecef, #dee2e6);
                padding: 20px;
                border-radius: 15px;
            }
            .stApp {
                background-color: #dee2e6;
            }
            .sidebar .sidebar-content {
                background-color: #343a40;
                color: white;
            }
            h1 {
                color: #0d6efd;
                font-weight: bold;
            }
            .css-18e3th9 {
                padding-top: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    st.title("Sentiment Analysis Dashboard with AI Chatbot 👁️🔍")
    st.write("Analyze sentiment of your text using TinyLlama!")

    st.sidebar.title("Settings")
    model_name = st.sidebar.selectbox("Select LLM Model", ["TinyLlama", "Phi-3", "Mixtral"])

    analyzer = get_sentiment_analyzer(model_name)

    # Text Input Area
    text_input = st.text_area("Enter text here for sentiment analysis:")

    # CSV Upload Option
    uploaded_file = st.file_uploader("Or Upload a CSV file for Batch Sentiment Analysis", type=["csv"])

    # Prepare list of texts to analyze
    texts_to_analyze = []
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            texts_to_analyze = df['text'].tolist()
            st.success(f"Uploaded {len(texts_to_analyze)} texts from CSV.")
        else:
            st.error("CSV must contain a column named 'text'.")
    elif text_input.strip() != "":
        texts_to_analyze = [text_input.strip()]

    if 'results_data' not in st.session_state:
        st.session_state.results_data = []

    if st.button("Analyze Sentiment"):
        if texts_to_analyze:
            st.session_state.results_data = []
            for idx, text in enumerate(texts_to_analyze, start=1):
                result = analyzer.analyze_sentiment(text)
                st.subheader(f"Result {idx}: {result['classification'].capitalize()}")
                st.plotly_chart(create_sentiment_gauge(result['confidence'], result['classification']))
                st.pyplot(create_word_cloud(text))
                st.session_state.results_data.append({
                    'Text': text,
                    'Sentiment': result['classification']
                })
        else:
            st.warning("Please enter text or upload a CSV to analyze.")

    # Download Results
    if st.session_state.results_data:
        results_df = pd.DataFrame(st.session_state.results_data)
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name='sentiment_results.csv',
            mime='text/csv'
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Created by **Ayan Dev Burman** with ❤️ using Streamlit & Ollama")

    # Sidebar Chat Assistant
    st.sidebar.title("💬 AI Chat Assistant")
    user_question = st.sidebar.text_input("Ask about Sentiment Results:", "")
    if user_question:
        if st.session_state.results_data:
            analyzed_data = "\n".join([f"{item['Text']} - {item['Sentiment']}" for item in st.session_state.results_data])
            answer = chat_about_sentiment(user_question, analyzed_data)
            st.sidebar.markdown("#### AI Answer:")
            st.sidebar.write(answer)
        else:
            st.sidebar.warning("Please analyze some text first!")

if __name__ == "__main__":
    main()

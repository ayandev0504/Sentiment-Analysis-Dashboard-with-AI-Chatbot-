# Sentiment Analysis Dashboard with AI Chatbot 👁️🔍

An interactive Streamlit dashboard to analyze sentiments from text or CSV files, visualize insights with gauges and word clouds, and chat with an AI-powered assistant (TinyLlama, Phi-3, Mixtral) for interactive feedback summarization.

---

## 🌟 Features

* Single Text & Batch CSV Sentiment Analysis
* Real-time Word Cloud Visualization
* Sentiment Confidence Gauge (Plotly)
* Downloadable Sentiment CSV Report
* **AI Chat Assistant Panel** for queries like:

  * "Summarize all positive feedback"
  * "Which text has the most negative sentiment?"
* Supports **TinyLlama, Phi-3, Mixtral** via Ollama backend.

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ayandev0504/Sentiment-Analysis-Dashboard-with-AI-Chatbot-.git

```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install & Run Ollama

```bash
ollama pull tinyllama
ollama pull phi-3
ollama pull mixtral
ollama serve
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

---

---

## 📃 Usage

* **Enter Single Text** in the input box and click "Analyze Sentiment".
* **Upload CSV** with a column named `text` for batch analysis.
* Download the analysis results as CSV.
* Use the **Chat Assistant Panel** to interact with AI.

---

## 📖 Example CSV Format

```csv
text
"I love this product, it's amazing!"
"Terrible service, not recommended."
"It's okay, nothing special."
```

---

## 🛡️ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ayan Dev Burman**
[GitHub](https://github.com/ayandev0504) | [LinkedIn](https://www.linkedin.com/in/ayan-dev-b-291805217/)

---

## 📈 Future Enhancements

* Sentiment Trend Over Time Visualization
* Multi-Language Sentiment Support
* AI-based Sentiment Correction Suggestions
* Deployment on Streamlit Cloud / HuggingFace Spaces

---

## 💌 Contributing

PRs are welcome! Feel free to fork this repo and suggest improvements.

---

## 🚀 Credits

* [Streamlit](https://streamlit.io)
* [LangChain](https://python.langchain.com)
* [Ollama](https://ollama.com)
* [NLTK](https://nltk.org)
* [Plotly](https://plotly.com)
* [WordCloud](https://amueller.github.io/word_cloud/)

---

## 💡 Demo Screenshot

![Dashboard Screenshot](path/to/your/screenshot.png)

---

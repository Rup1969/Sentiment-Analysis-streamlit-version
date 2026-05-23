# 🧠 Sentiment Check: Advanced Neural Emotion Detection System

A production-ready Natural Language Processing (NLP) web application that detects multiple simultaneous emotional states in text. Unlike simple binary positive/negative classifiers, this system performs **multi-label classification** across five distinct emotional spectrums: Anger, Fear, Joy, Sadness, and Surprise.

The application features an enterprise-ready, document-style layout built using Streamlit and is powered by a custom fine-tuned RoBERTa transformer architecture.

---

## 🚀 Live Core Features

* **Multi-Label Probability Analysis:** Captures the true complexity of human language by identifying multiple overlapping emotions (e.g., a sentence displaying high anxiety and resistance simultaneously).
* **Sorted Spectrum Reporting:** Automatically aggregates and arranges emotional metrics in descending order, displaying the most dominant emotional markers right at the top.
* **Contextual Exception Adjustments:** Features a custom backend penalty matrix to mitigate specific language biases (such as contextual classification adjustments for terms like "sleepy", "tired", or "hungry").
* **Optimized Local CPU Architecture:** Built with memory-efficient pipeline wrappers (`low_cpu_mem_usage=True`) to run complex neural networks instantly on shared, free cloud hardware or a standard local CPU.

---

## 🛠️ Architecture & Core Dependencies

The machine learning core uses the Hugging Face `transformers` repository structure, pulling fine-tuned weights directly into a tokenized PyTorch evaluation layer.

* **Frontend Framework:** Streamlit (Custom Form Layout)
* **Deep Learning Engine:** PyTorch (`torch`)
* **Transformer Pipeline:** Hugging Face `transformers` (Tokenizer & Sequence Classification)
* **Model Base:** Fine-Tuned RoBERTa Configuration (`rup69/Sentiment-Analysis`)

---

## 💻 Local Installation & Setup

If you want to run this application locally on your machine, ensure you have Python 3.8+ installed, then follow these simple steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Rup1969/Sentiment-Analysis-streamlit-version.git](https://github.com/Rup1969/Sentiment-Analysis-streamlit-version.git)
   cd Sentiment-Analysis-streamlit-version

# Install Core Requirements:
pip install -r requirements.txt
# Launch the Streamlit Interface:
streamlit run streamlit_app.py

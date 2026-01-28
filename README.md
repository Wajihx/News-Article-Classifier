# 📰 News Article Classifier

An NLP-powered web application that automatically classifies news articles into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

## 🚀 Features

- **Multiple Input Methods**: Paste text, upload PDF/TXT files, or test with sample articles
- **Fine-tuned DistilBERT Model**: Trained on AG News dataset for accurate classification
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Real-time Analysis**: Get instant predictions with confidence scores

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Streamlit

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Wajih_Esghayri_IAOC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
.
├── app.py                      # Streamlit web application
├── utils.py                    # Model loading and helper functions
├── requirements.txt            # Python dependencies
├── news/                       # Sample news articles
└── colab_stuff/training/       # Model checkpoints and results
```

## 🎯 Model

The classifier uses a fine-tuned **DistilBERT** model trained on the AG News dataset, achieving high accuracy across all four news categories.

## 👤 Author

**Wajih Esghayri**  
NLP Project - Master S3

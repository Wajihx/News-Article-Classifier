# Libraries
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import PyPDF2
import re


# Load the trained model and tokenizer once
CHECKPOINT = "./colab_stuff/ttt/results"

model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT)
tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT)
model.eval()


# Label mapping for news categories
LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Emoji mapping for visual appeal
LABEL_EMOJIS = {
    0: "🌍",
    1: "⚽",
    2: "💼",
    3: "🔬"
}


def read_pdf(file_input):
    """Extract text from PDF - handles both file paths and file-like objects"""
    text = ""
    reader = PyPDF2.PdfReader(file_input)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return clean_text(text)


def clean_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()


def predict_news_article(text, max_words=1000):
    """
    Classify news article into one of four categories: World, Sports, Business, or Sci/Tech.
    
    Args:
        text: The article text to classify
        max_words: Maximum number of words to use for classification (default: 1000)
    
    Returns:
        pred_label: Predicted category ID (0-3)
        confidence: Confidence score (0-1)
        classified_text: The text snippet used for classification
    """
    model.eval()
    
    # Use first N words of the article
    words = text.split()
    text_snippet = " ".join(words[:max_words])
    
    with torch.no_grad():
        inputs = tokenizer(
            text_snippet,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1)[0][pred].item()
    
    return pred, confidence, text_snippet
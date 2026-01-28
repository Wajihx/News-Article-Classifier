import streamlit as st
from utils import read_pdf, predict_news_article, LABELS
import os

# Page configuration
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide"
)

# Main layout - 2 columns
left_col, right_col = st.columns([1, 1])

# LEFT COLUMN - About and Input Methods
with left_col:
    st.title("📰 News Article Classifier")    
    st.markdown("### About")
    st.write("""
    This platform uses a **NLP AI model** to automatically classify news articles into categories.
    Simply choose an input method below to get started.
    """)
    
    st.info("**Available Categories:** World | Sports | Business | Sci/Tech")
    
    st.markdown("---")
    
    st.markdown("### Input Method")
    
    # Three input options as tabs
    tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "📤 Upload File", "📁 Test on Our News"])
    
    article_text = None
    
    with tab1:
        pasted_text = st.text_area(
            "Paste your news article here:",
            height=275,
            placeholder="Enter or paste the news article text..."
        )
        if pasted_text and pasted_text.strip():
            article_text = pasted_text
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"]
        )
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                article_text = read_pdf(uploaded_file)
            else:
                article_text = str(uploaded_file.read(), "utf-8")
            st.success(f"✅ Loaded: {uploaded_file.name}")
    
    with tab3:
        st.write("Select a sample news article to test the classifier:")
        if os.path.exists("./news"):
            papers = [f for f in os.listdir("./news") if f.endswith(('.txt', '.pdf'))]
            if papers:
                selected = st.selectbox("Choose a sample:", papers)
                if selected:
                    file_path = os.path.join("./news", selected)
                    if selected.endswith(".pdf"):
                        article_text = read_pdf(file_path)
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            article_text = f.read()
                    st.success(f"✅ Loaded: {selected}")
            else:
                st.warning("No sample files found.")
        else:
            st.warning("News folder not found.")


# RIGHT COLUMN - Results and Analysis
with right_col:
    st.markdown("### Results")
    
    if article_text and article_text.strip():
        st.markdown("---")
        
        words_count = len(article_text.split())
        chars_count = len(article_text)
        
        col1, col2 = st.columns(2)
        col1.metric("Words", f"{words_count:,}")
        col2.metric("Characters", f"{chars_count:,}")
        
        with st.expander("Show article preview"):
            preview_length = min(1000, len(article_text))
            st.text_area("", article_text[:preview_length] + ("..." if len(article_text) > 1000 else ""), height=150, disabled=True, label_visibility="collapsed")
        
        st.markdown("---")
        
        # Classify button
        if st.button("🚀 Classify Article", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                pred_label, confidence, classified_text = predict_news_article(article_text)
                category_name = LABELS[pred_label]
                
                # Display result
                st.success("✅ Classification Complete!")
                
                # Results in horizontal layout
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("Category", category_name)
                
                with res_col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with res_col3:
                    st.metric("Class ID", pred_label)
                
                st.progress(confidence)
                
                with st.expander("View classification details"):
                    st.write(f"**Category:** {category_name}")
                    st.write(f"**Class ID:** {pred_label}")
                    st.write(f"**Confidence:** {confidence:.2%}")
                    st.write(f"**Words analyzed:** {len(classified_text.split())}")
                    st.write(f"**Model:** DistilBERT fine-tuned on AG News dataset")
                
                with st.expander("View analyzed text"):
                    st.text_area("", classified_text, height=200, disabled=True, label_visibility="collapsed")
    
    else:
        st.info("👈 Please provide a news article using one of the input methods on the left.")

# Footer
st.markdown("---")
st.caption("Powered by DistilBERT & Streamlit")
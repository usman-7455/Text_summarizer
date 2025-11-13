import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load fine-tuned summarization model from Hugging Face
@st.cache_resource
def load_model():
    model_name = "zentom/summary_generator_T5_Finetuned"  # Your HF model name
    
    try:
        # Load model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

# Streamlit UI
st.title(" T5 Fine-Tuned Text Summarizer")
st.write("Paste any article or long text below and get a concise summary using our fine-tuned T5 model.")

user_input = st.text_area("Enter your text here:", height=200, 
                         placeholder="Paste your article, blog post, or any long text here...")

col1, col2 = st.columns(2)

with col1:
    max_len = st.slider("Maximum summary length", min_value=50, max_value=300, value=120)

with col2:
    min_len = st.slider("Minimum summary length", min_value=20, max_value=100, value=30)

# Add some info
with st.expander("ℹ About this model"):
    st.write("""
    This model is a fine-tuned version of **T5-small** trained on the CNN/DailyMail dataset.
    - **Base Model**: T5-small
    - **Training Data**: CNN/DailyMail articles
    - **Purpose**: Text summarization
    - **Input Format**: The model automatically adds 'summarize: ' prefix
    """)

if st.button(" Generate Summary", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to summarize.")
    elif model is None or tokenizer is None:
        st.error(" Model failed to load. Please check the console for errors.")
    else:
        try:
            with st.spinner(" Generating summary... This may take a few seconds."):
                # Prepare input for T5 (add the 'summarize: ' prefix)
                input_text = "summarize: " + user_input
                
                # Tokenize
                inputs = tokenizer.encode(
                    input_text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                )
                
                # Generate summary
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=max_len,
                        min_length=min_len,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        repetition_penalty=1.2  # Reduce repetition
                    )
                
                # Decode
                summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Display results
            st.success("✅ Summary Generated!")
            
            st.subheader("📄 Generated Summary")
            st.info(summary_text)
            
            # Original text preview
            st.subheader("📜 Original Text Preview")
            st.text_area("", value=user_input[:500] + "..." if len(user_input) > 500 else user_input, 
                        height=150, key="original_preview", label_visibility="collapsed")
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Length", f"{len(user_input)} chars")
            with col2:
                st.metric("Summary Length", f"{len(summary_text)} chars")
            with col3:
                st.metric("Compression Ratio", f"{len(summary_text)/len(user_input)*100:.1f}%")

        except Exception as e:
            st.error(f" Error during summarization: {str(e)}")


import streamlit as st
import torch
from transformers import pipeline
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_classifier():
    # Use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", 
                    model="google/vit-base-patch16-224", 
                    device=device)

classifier = load_classifier()

# --- UI Header ---
st.title("🖼️ Smart Image Classifier")
st.markdown("Upload an image, and the Vision Transformer (ViT) will identify the objects within it.")

# --- Sidebar / Settings ---
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=3)

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns: one for the image, one for results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner('Analyzing...'):
            # Perform inference
            results = classifier(image, top_k=top_k)
            
            st.subheader("Classification Results")
            
            # Display results as progress bars for a better UI
            for res in results:
                label = res['label']
                score = res['score']
                
                st.write(f"**{label.title()}**")
                st.progress(score)
                st.write(f"Confidence: {score:.2%}")
                st.divider()

else:
    st.info("Please upload an image to start the classification.")


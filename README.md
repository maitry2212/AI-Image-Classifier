# AI Image Classifier

A clean, high-performance web application built with **Streamlit** and **Hugging Face Transformers**. This app utilizes Google's **Vision Transformer (ViT)** to classify images into over 1,000 categories with real-time confidence scoring.

## ✨ Features

* **Instant Inference:** Drag and drop images for immediate classification.
* **Visual Confidence:** Results are displayed using interactive progress bars.
* **Adjustable Sensitivity:** Sidebar slider to control how many top-ranking labels are shown ().
* **Resource Efficient:** Implements `@st.cache_resource` for fast model loading and memory management.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Deep Learning:** [PyTorch](https://pytorch.org/)
* **Model Hub:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* **Pre-trained Model:** `google/vit-base-patch16-224`

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/image-classifier-vit.git
cd image-classifier-vit

```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```bash
pip install -r requirements.txt

```

*Note: Your `requirements.txt` should contain:*

```text
streamlit
transformers
torch
pillow

```

### 3. Run the App

```bash
streamlit run app.py

```

---

## 📖 How It Works

The app uses the **Vision Transformer (ViT)**, which treats an image as a sequence of patches (similar to how BERT treats words in a sentence).

1. **Preprocessing:** The image is resized to  pixels and normalized.
2. **Patch Embedding:** The image is split into  patches.
3. **Transformer Encoder:** The model processes these patches to understand global context and object features.
4. **Classification:** The final hidden state is passed to a linear layer to predict the object class.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for features (like batch processing or more model options), feel free to open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

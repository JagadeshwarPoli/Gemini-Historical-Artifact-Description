<img width="509" height="836" alt="image" src="https://github.com/user-attachments/assets/10cdec89-656f-4c75-a4cd-584a482807e5" /># 🏺 AI Historical Artifact Description App

An AI-powered Streamlit web application that generates detailed historical descriptions of artifacts from uploaded images using transformer-based vision and language models.

---

## 📌 Project Overview

This project analyzes an uploaded image of a historical artifact and automatically generates a structured, detailed historical interpretation including:

- Name of the artifact
- Possible civilization or origin
- Approximate time period
- Cultural and historical significance
- Artistic and symbolic interpretation
- Interesting historical insights

The system runs completely offline using pre-trained transformer models.

---

## 🧠 AI Architecture

The application uses a two-stage AI pipeline:

### 1️⃣ Image Understanding (Vision Model)
- **Model:** `Salesforce/blip-image-captioning-base`
- Extracts a visual caption from the uploaded artifact image.

### 2️⃣ Historical Reasoning (Language Model)
- **Model:** `google/flan-t5-large`
- Expands the visual caption into a detailed historical analysis.
- Generates minimum 10+ sentences with contextual interpretation.

---

## 🛠 Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- PIL (Image Processing)

---

## 🚀 Features

- Upload artifact image (JPG, JPEG, PNG)
- Automatic image caption generation
- AI-generated historical analysis (150+ words)
- Structured, professional output
- Fully offline (no API key required)
- Cached model loading for performance

## ▶ How To Run The Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/JagadeshwarPoli/Gemini-Historical-Artifact-Description.git
cd Gemini-Historical-Artifact-Description

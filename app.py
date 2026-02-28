import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration
)

st.set_page_config(page_title="AI Historical Artifact Description App")

st.title("🏺 AI Historical Artifact Description App")
st.write("Upload an image of a historical artifact")

# ----------------------------
# Load Models (Cached)
# ----------------------------

@st.cache_resource
def load_models():
    # Image Captioning Model
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # Text Enhancement Model (Stronger reasoning)
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large"
    )

    return blip_processor, blip_model, tokenizer, t5_model


blip_processor, blip_model, tokenizer, t5_model = load_models()

# ----------------------------
# Upload Image
# ----------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Generate Artifact Description"):
        with st.spinner("Analyzing artifact..."):

            # Step 1: Generate caption from image
            inputs = blip_processor(image, return_tensors="pt")
            output = blip_model.generate(**inputs)
            caption = blip_processor.decode(
                output[0],
                skip_special_tokens=True
            )

            # Step 2: Expand into detailed historical paragraph
            prompt = f"""
You are a professional historian and archaeologist.

The image caption is:
"{caption}"

Write a detailed historical description of this artifact.

Requirements:
- Minimum 10 complete sentences
- At least 150 words
- Provide realistic historical interpretation
- Explain cultural context
- Mention possible civilization
- Mention approximate time period
- Explain artistic or symbolic meaning
- Discuss historical importance

Write in paragraph format.
Do NOT repeat the caption directly.
"""

            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            ).input_ids

            generated_ids = t5_model.generate(
                input_ids,
                max_length=400,
                min_length=200,
                num_beams=5,
                temperature=0.8,
                early_stopping=True
            )

            detailed_text = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            st.subheader("📜 AI Generated Historical Description")
            st.write(detailed_text)
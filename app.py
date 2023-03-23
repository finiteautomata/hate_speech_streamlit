"""
Streamlit for classification using transformers
"""
# Import streamlit
import streamlit as st
import torch
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "piubamas/beto-contextualized-hate-speech"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Horrible pero efectivo
id2label = [model.config.id2label[k] for k in range(len(model.config.id2label))]

def predict(*args):
    args = [preprocess_tweet(arg) for arg in args]
    encoding = tokenizer.encode_plus(*args)

    inputs = {
        k:torch.LongTensor(encoding[k]).reshape(1, -1) for k in {"input_ids", "attention_mask", "token_type_ids"}
    }

    output = model.forward(
        **inputs
    )

    chars = list(zip(id2label, list(output.logits[0].detach().cpu().numpy() > 0)))

    return [char for char, pred in chars if pred]



# Create a title
st.title("Detección de discurso de odio en medios")

# Create an input box for context
context = st.text_input("Contexto", "China prohíbe la cría de perros para consumo humano")
# Create an input box for text
text = st.text_input("Comentario", "Chinos hdrmp hay que matarlos a todos")

# Create a button to classify
if st.button("Predict"):
    # Classify the text
    prediction = predict(text, context)
    # Print the classification
    st.write(prediction)


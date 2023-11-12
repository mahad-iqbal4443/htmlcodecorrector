import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Function to load the model and tokenizers
def load_model_and_tokenizers():
    # Load tokenizer_X
    with open('tokenizer_X_config.pickle', 'rb') as handle:
        tokenizer_X_config = pickle.load(handle)

    loaded_tokenizer_X = Tokenizer()
    loaded_tokenizer_X.word_index = tokenizer_X_config['word_index']
    loaded_tokenizer_X.document_count = tokenizer_X_config['document_count']
    loaded_tokenizer_X.char_level = tokenizer_X_config['char_level']
    loaded_tokenizer_X.oov_token = tokenizer_X_config['oov_token']

    # Load tokenizer_Y
    with open('tokenizer_y_config.pickle', 'rb') as handle:
        tokenizer_Y_config = pickle.load(handle)

    loaded_tokenizer_Y = Tokenizer()
    loaded_tokenizer_Y.word_index = tokenizer_Y_config['word_index']
    loaded_tokenizer_Y.document_count = tokenizer_Y_config['document_count']
    loaded_tokenizer_Y.char_level = tokenizer_Y_config['char_level']
    loaded_tokenizer_Y.oov_token = tokenizer_Y_config['oov_token']

    # Load the model
    loaded_model = tf.keras.models.load_model('html_correction_model.h5')

    return loaded_model, loaded_tokenizer_X, loaded_tokenizer_Y

# Function to correct HTML text
def correct_html(input_text, loaded_model, loaded_tokenizer_X, loaded_tokenizer_Y):
    # Tokenize the input text
    input_seq = loaded_tokenizer_X.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=X_pad.shape[1])

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_pad)
    predicted_seq = np.argmax(predictions, axis=-1)[0]

    # Convert the predicted sequence back to text using the loaded tokenizer
    predicted_text = loaded_tokenizer_Y.sequences_to_texts([predicted_seq])[0]

    return predicted_text

# Load the model and tokenizers
model, tokenizer_X, tokenizer_Y = load_model_and_tokenizers()

# Streamlit app
st.title("HTML Correction Model")
st.markdown("### Correct your HTML markup with this model!")

# Input text box for HTML input
input_text = st.text_area("Enter HTML Text:", "<em>This is emphasized text</em>")

# Button to trigger correction
if st.button("Correct HTML"):
    # Correct the HTML text
    corrected_text = correct_html(input_text, model, tokenizer_X, tokenizer_Y)

    # Display corrected text
    st.markdown("### Corrected HTML:")
    st.code(corrected_text, language="html")

# Display model accuracy (replace with the actual accuracy value)
model_accuracy = 0.85
st.sidebar.markdown("### Model Accuracy")
st.sidebar.info(f"The model accuracy is: {model_accuracy:.2%}")

# Add some styling
st.markdown(
    """
    <style>
        body {
            color: #333;
            background-color: #f8f9fa;
        }
        .stTextInput {
            color: #495057;
            background-color: #fff;
            border: 1px solid #ced4da;
            border-radius: .2rem;
        }
        .stButton {
            color: #fff;
            background-color: #007bff;
            border-color: #007bff;
        }
        .stButton:hover {
            background-color: #0056b3;
            border-color: #0056b3;
        }
        .stMarkdown {
            color: #495057;
        }
        .stSidebar {
            background-color: #f8f9fa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

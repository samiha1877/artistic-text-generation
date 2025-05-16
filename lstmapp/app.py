# poetry_generator_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
from collections import defaultdict
import json
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# ----------------------------
# 1. Streamlit Page Configuration
# ----------------------------

st.set_page_config(
    page_title="🎨 AI Poetry Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="auto",
)

# ----------------------------
# 2. Define Word Mappings
# ----------------------------

@st.cache_data
def load_word_mappings():
    """
    Loads the word2idx and idx2word mappings from JSON files.

    Returns:
        word2idx (dict): Mapping from words to indices.
        idx2word (dict): Mapping from indices to words.
    """
    try:
        with open('word2idx.json', 'r', encoding='utf-8') as f:
            word2idx = json.load(f)
        with open('idx2word.json', 'r', encoding='utf-8') as f:
            idx2word_str_keys = json.load(f)
            # Convert string keys back to integers
            idx2word = {int(idx): word for idx, word in idx2word_str_keys.items()}
        logger.info("Word mappings loaded successfully.")
        return word2idx, idx2word
    except FileNotFoundError:
        st.error("word2idx.json or idx2word.json file not found. Please ensure they exist in the app directory.")
        return {}, {}
    except Exception as e:
        st.error(f"Error loading word mappings: {e}")
        return {}, {}

word2idx, idx2word = load_word_mappings()

# If mappings are not loaded, stop the app
if not word2idx or not idx2word:
    st.stop()

# ----------------------------
# 3. Load the Model
# ----------------------------

@st.cache_resource
def load_model():
    """
    Loads the trained TensorFlow model from the specified path.

    Returns:
        model (tf.keras.Model): Loaded TensorFlow model.
    """
    try:
        model = tf.keras.models.load_model('poetry_generation_model.keras')
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# If model is not loaded, stop the app
if model is None:
    st.stop()

# ----------------------------
# 4. Define Text Generation Function
# ----------------------------

def generate_text(model, start_string, word2idx, idx2word, num_generate=50, temperature=1.0, top_k=5):
    """
    Generates text using the trained model.

    Parameters:
    - model: Trained TensorFlow model.
    - start_string: The initial string to start the generation.
    - word2idx: Dictionary mapping words to indices.
    - idx2word: Dictionary mapping indices to words.
    - num_generate: Number of words to generate.
    - temperature: Controls the randomness of predictions. Lower values make the text more predictable.
    - top_k: Number of top predictions to consider for each step.

    Returns:
    - Generated text as a string.
    """
    try:
        # Tokenize the start string
        input_words = nltk.word_tokenize(start_string.lower())
        input_indices = [word2idx.get(word, word2idx.get("<UNK>", 1)) for word in input_words]
        input_eval = tf.expand_dims(input_indices, 0)

        text_generated = []

        for _ in range(num_generate):
            predictions = model(input_eval)
            predictions = predictions[:, -1, :] / temperature
            predictions_np = predictions.numpy()[0]

            # Exclude <UNK> token from top predictions
            unk_index = word2idx.get("<UNK>", 1)
            if unk_index < len(predictions_np):
                predictions_np[unk_index] = -np.inf  # Set probability of <UNK> to negative infinity

            # Top-K sampling
            top_k_indices = predictions_np.argsort()[-top_k:]
            top_k_probs = predictions_np[top_k_indices]
            top_k_probs = top_k_probs - np.max(top_k_probs)  # For numerical stability
            top_k_probs = np.exp(top_k_probs)
            top_k_probs = top_k_probs / np.sum(top_k_probs)  # Normalize

            # Select the next word
            selected_id = np.random.choice(top_k_indices, p=top_k_probs)

            # Append the word
            generated_word = idx2word.get(selected_id, "")
            if generated_word == "":
                break  # Stop if no valid word is found
            text_generated.append(generated_word)

            # Update the input
            input_eval = tf.concat([input_eval, tf.expand_dims([selected_id], 0)], axis=1)

        return start_string + ' ' + ' '.join(text_generated)
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return ""

# ----------------------------
# 5. Streamlit App Layout
# ----------------------------

# App Title
st.title("🎨 AI-Powered Poetry Generator")

# App Description
st.markdown("""
Welcome to the **AI Poetry Generator**! This application uses a trained neural network model to generate poetic verses based on your input.

**How to Use:**
1. Enter a starting phrase or sentence.
2. Choose the number of words you want to generate.
3. Adjust the temperature to control the creativity of the output.
4. Click on the "Generate Poetry" button to see your AI-generated poem!

*Enjoy creating beautiful poetry with the power of AI!*
""")

# Sidebar for Inputs
st.sidebar.header("Customize Your Poem")

# User Inputs
start_string = st.sidebar.text_input(
    "🔤 Starting Phrase",
    value="Once upon a midnight dreary",
    help="Enter the initial words to start the poem."
)

num_generate = st.sidebar.slider(
    "🔢 Number of Words to Generate",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Select how many words you want the AI to generate."
)

temperature = st.sidebar.slider(
    "🎚️ Creativity Level (Temperature)",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.1,
    help="Adjust the randomness of the generated text. Lower values make the output more predictable."
)

top_k = st.sidebar.slider(
    "🔝 Top-K Sampling",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Limits the next word choices to the top K probable words to make the generation more coherent."
)

# Generate Button
generate_button = st.sidebar.button("✨ Generate Poetry")

# Display Area
st.subheader("🖋️ Generated Poem")
if generate_button:
    if start_string.strip() == "":
        st.warning("Please enter a starting phrase to generate poetry.")
    else:
        with st.spinner("Generating your poem..."):
            generated_poem = generate_text(
                model=model,
                start_string=start_string,
                word2idx=word2idx,
                idx2word=idx2word,
                num_generate=num_generate,
                temperature=temperature,
                top_k=top_k
            )
        if generated_poem:
            # Post-processing to clean up <UNK> tokens if any
            generated_poem = generated_poem.replace("<UNK>", "").strip()
            st.write(generated_poem)
        else:
            st.error("Failed to generate poem. Please try different settings.")
else:
    st.info("Enter your preferences in the sidebar and click on 'Generate Poetry' to begin.")

# ----------------------------
# 6. Aesthetic Enhancements
# ----------------------------

# Add custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #1E1E1E;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1E1E1E;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2D2D2D;
    }
    
    /* Text colors */
    .stMarkdown {
        color: #FFFFFF;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Button styling */
    .stButton>button {
        color: white;
        background-color: #2e7bcf;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #3D3D3D;
        color: white;
    }
    
    /* Slider background */
    .stSlider>div>div>div {
        background-color: #3D3D3D;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #2D2D2D;
        color: white;
    }
    
    /* Generated text area */
    .stTextArea>div>div>textarea {
        background-color: #2D2D2D;
        color: white;
    }
    
    /* Remove any white backgrounds from widgets */
    .element-container {
        background-color: transparent !important;
    }
    
    .stSpinner>div {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

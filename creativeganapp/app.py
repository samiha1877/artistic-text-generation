# poetry_generator_app.py

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
import os
from typing import Tuple

# -----------------------------------
# 1. Discriminator Model Definition
# -----------------------------------

class Discriminator(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(Discriminator, self).__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name).transformer
        self.discriminator_head = nn.Sequential(
            nn.Linear(self.transformer.config.n_embd, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
            # No Sigmoid here since we'll use BCEWithLogitsLoss
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        last_hidden = hidden_states[:, -1, :]      # (batch_size, hidden_size)
        logits = self.discriminator_head(last_hidden)  # (batch_size, 1)
        return logits

# -----------------------------------
# 2. Load Models with Caching
# -----------------------------------

@st.cache_resource
def load_generator(model_path: str) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Load the generator model and tokenizer.
    """
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        generator = GPT2LMHeadModel.from_pretrained(model_path)
        generator.to(device)
        generator.eval()
        return generator, tokenizer
    except Exception as e:
        st.error(f"Error loading generator model: {e}")
        st.stop()

@st.cache_resource
def load_discriminator(model_path: str, device: torch.device) -> Discriminator:
    """
    Load the discriminator model.
    """
    try:
        discriminator = Discriminator()
        discriminator.load_state_dict(torch.load(model_path, map_location=device))
        discriminator.to(device)
        discriminator.eval()
        return discriminator
    except Exception as e:
        st.error(f"Error loading discriminator model: {e}")
        st.stop()

# -----------------------------------
# 3. Utility Functions
# -----------------------------------

def generate_poem(
    generator: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int,
    creativity_level: float,
    diversity_limit: int,
    probability_threshold: float
) -> str:
    """
    Generate a poem based on the prompt and generation parameters.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape).to(device)

    try:
        output_ids = generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=creativity_level,
            top_k=diversity_limit,
            top_p=probability_threshold,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return ""

def evaluate_poem(
    discriminator: Discriminator,
    tokenizer: GPT2Tokenizer,
    poem: str
) -> float:
    """
    Evaluate the generated poem using the discriminator.
    Returns the probability of the poem being real.
    """
    inputs = tokenizer.encode_plus(
        poem,
        add_special_tokens=True,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = discriminator(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)
        prob_real = probabilities.item()
    return prob_real

# -----------------------------------
# 4. Streamlit App Layout
# -----------------------------------

# Set page configuration
st.set_page_config(
    page_title="Creative GAN Poetry Generator",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto"
)

# Title and Description
st.title("📝 Creative GAN for Poetry Generation")
st.markdown("""
Welcome to the **Creative GAN for Poetry and Prose Generation** app! 
Generate beautiful poems based on your prompts and evaluate their authenticity.
""")

# Sidebar for Inputs
st.sidebar.header("🎨 Generation Parameters")

# User Input: Prompt
prompt = st.sidebar.text_input(
    "✏️ Enter a prompt for your poem:",
    value="The serene beauty of the night sky"
)

# Collapsible section for Advanced Settings
with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    # Generation Parameters with Descriptive Labels and Explanations
    creativity_level = st.slider(
        "🎭 Creativity Level",
        min_value=0.5,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="""
        Controls the randomness of the generated text. 
        Higher values make the output more creative and diverse, 
        while lower values make it more focused and deterministic.
        """
    )
    
    diversity_limit = st.slider(
        "🔍 Diversity Limit (Top-k)",
        min_value=0,
        max_value=100,
        value=50,
        step=10,
        help="""
        Limits the sampling pool to the top 'k' most probable next words.
        Lower values make the text more predictable, while higher values increase diversity.
        """
    )
    
    probability_threshold = st.slider(
        "📊 Probability Threshold (Top-p)",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="""
        Ensures that the cumulative probability of the next word is below this threshold.
        Balances between diversity and coherence in the generated text.
        """
    )
    
    max_new_tokens = st.slider(
        "📝 Maximum Length of Generated Text",
        min_value=50,
        max_value=300,
        value=100,
        step=10,
        help="""
        Specifies the maximum number of new tokens (words/pieces) to generate.
        Longer texts provide more content but may take more time to generate.
        """
    )
    
# Button to Generate Poem
generate_button = st.sidebar.button("🚀 Generate Poetry")

# -----------------------------------
# 5. Load Models
# -----------------------------------

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to models
generator_path = "saved_models/generator"  # Adjust if necessary
discriminator_path = "saved_models/discriminator.pth"  # Adjust if necessary

# Load models
with st.spinner("🔄 Loading models..."):
    generator, tokenizer = load_generator(generator_path)
    discriminator = load_discriminator(discriminator_path, device)

st.success("✅ Models loaded successfully!")

# -----------------------------------
# 6. Generate and Display Poem
# -----------------------------------

if generate_button:
    if not prompt.strip():
        st.warning("⚠️ Please enter a valid prompt to generate a poetry.")
    else:
        with st.spinner("🖋️ Generating your poetry..."):
            poem = generate_poem(
                generator=generator,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                creativity_level=creativity_level,
                diversity_limit=diversity_limit,
                probability_threshold=probability_threshold
            )
        
        if poem:
            st.markdown("### 🖋️ Generated Poetry")
            st.write(poem)
            
            # Evaluate the poem
            with st.spinner("🔍 Evaluating the poem's authenticity..."):
                prob_real = evaluate_poem(discriminator, tokenizer, poem)
            
            st.markdown("### 🧐 Discriminator Evaluation")
            st.write(f"**Probability of being real:** {prob_real*100:.2f}%")
            if prob_real > 0.5:
                st.success("✅ The discriminator considers this poetry to be **real**.")
            else:
                st.warning("⚠️ The discriminator considers this poetry to be **fake**.")



# poetry_app.py

import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import numpy as np
import os

# Suppress warnings (optional)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the RNN model with Attention and Batch Normalization
class PoetryRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx):
        super(PoetryRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # BatchNorm for Embedding Layer
        self.batch_norm_emb = nn.BatchNorm1d(embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.3,  
            bidirectional=True
        )
        
        # BatchNorm for LSTM Output
        self.batch_norm_lstm = nn.BatchNorm1d(hidden_dim * 2)
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  
        
        # Apply BatchNorm to Embedding
        batch_size, seq_length, embedding_dim = embedded.size()
        embedded = embedded.view(-1, embedding_dim)  
        embedded = self.batch_norm_emb(embedded)
        embedded = embedded.view(batch_size, seq_length, embedding_dim)  
        
        lstm_out, _ = self.lstm(embedded)  
        
        # Apply BatchNorm to LSTM Output
        lstm_out_reshaped = lstm_out.contiguous().view(-1, lstm_out.shape[2])  
        lstm_out_norm = self.batch_norm_lstm(lstm_out_reshaped)
        lstm_out = lstm_out_norm.view(batch_size, seq_length, -1)  
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  
        context = torch.sum(attn_weights * lstm_out, dim=1)  
        
        context = self.dropout(context)
        out = self.fc(context)  
        return out

# Load the tokenizer
@st.cache_resource
def load_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    tokenizer.enable_truncation(max_length=512)
    return tokenizer

# Load the model
@st.cache_resource
def load_model(model_path, _tokenizer, vocab_size, embedding_dim, hidden_dim, num_layers, pad_id):
    model = PoetryRNN(vocab_size, embedding_dim, hidden_dim, num_layers, pad_id)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Generate poetry function
def generate_poetry(model, tokenizer, seed_text, max_length=100, seq_length=100, pad_id=0, temperature=1.0):
    tokens = tokenizer.encode(seed_text).ids
    generated = tokens.copy()
    
    for _ in range(max_length):
        # Prepare the input sequence
        if len(generated) >= seq_length:
            input_seq = generated[-seq_length:]
        else:
            input_seq = [pad_id] * (seq_length - len(generated)) + generated
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_seq)  # [1, vocab_size]
            logits = output / temperature
            probs = nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
            next_token = np.random.choice(len(probs), p=probs)
            if next_token == pad_id:
                break
            generated.append(next_token)
    
    generated_text = tokenizer.decode(generated)
    return generated_text

# Streamlit App
def main():
    st.set_page_config(page_title="🎨 Poetry Generator", layout="wide")
    st.title("🎨 Poetry Generator")
    
    
    # Sidebar for generation settings
    st.sidebar.header("Generation Settings")
    max_length = st.sidebar.slider("Max Generation Length", min_value=50, max_value=500, value=100, step=50)
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    
    # Input area
    seed_text = st.text_input("Enter your seed text here:", "the moon shines")
    
    if st.button("Generate Poetry"):
        if seed_text.strip() == "":
            st.warning("Please enter some seed text to generate poetry.")
        else:
            with st.spinner("Generating poetry..."):
                try:
                    # Load tokenizer and model
                    tokenizer = load_tokenizer("bpe_tokenizer.json")
                    
                    
                    VOCAB_SIZE = 10000 
                    EMBEDDING_DIM = 256
                    HIDDEN_DIM = 512
                    NUM_LAYERS = 2
                    pad_id = tokenizer.token_to_id("<pad>")
                    
                    model = load_model(
                        model_path="poetry_rnn_model.pth",
                        _tokenizer=tokenizer,  
                        vocab_size=VOCAB_SIZE,
                        embedding_dim=EMBEDDING_DIM,
                        hidden_dim=HIDDEN_DIM,
                        num_layers=NUM_LAYERS,
                        pad_id=pad_id
                    )
                    
                    # Generate poetry
                    generated_poem = generate_poetry(
                        model=model,
                        tokenizer=tokenizer,
                        seed_text=seed_text,
                        max_length=max_length,
                        seq_length=100,
                        pad_id=pad_id,
                        temperature=temperature
                    )
                    
                    st.success("🎉 Poetry Generated!")
                    st.text_area("Your Generated Poetry:", generated_poem, height=300)
                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    main()

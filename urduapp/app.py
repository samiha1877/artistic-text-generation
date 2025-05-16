# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm
import numpy as np
import math
from torch.nn.functional import log_softmax
from collections import deque

# ----------------------------
# Define the Transformer Decoder-Only Model
# ----------------------------

class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, mask):
        # x: [seq_len, batch_size, hidden_dim]
        # mask: [seq_len, seq_len]
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, ff_dim, dropout=0.3):
        super(DecoderOnlyTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(1000, embedding_dim)  # Supports up to 1000 tokens
        
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Ensure embedding_dim == hidden_dim
        assert embedding_dim == hidden_dim, "Embedding dimension must be equal to hidden dimension"
    
    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on the diagonal."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        # Generate mask
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)  # [seq_len, seq_len]
        
        # Token and positional embeddings
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, embedding_dim]
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)  # [batch_size, seq_len, embedding_dim]
        
        x = self.dropout(token_emb + pos_emb)  # [batch_size, seq_len, embedding_dim]
        
        # Transpose for MultiheadAttention: [seq_len, batch_size, hidden_dim]
        x = x.transpose(0, 1)
        
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.layer_norm(x)  # [seq_len, batch_size, hidden_dim]
        
        # Transpose back: [batch_size, seq_len, hidden_dim]
        x = x.transpose(0, 1)
        
        logits = self.fc_out(x)  # [batch_size, seq_len, vocab_size]
        
        # For prediction of the next token, we take the last token's logits
        logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        return logits

# ----------------------------
# Load the Trained Model and Tokenizer
# ----------------------------

@st.cache_resource
def load_model():
    # Define hyperparameters (must match the training configuration)
    config = {
        'vocab_size': 8000,  # Adjust if different
        'embedding_dim': 512,
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'ff_dim': 2048,
        'dropout': 0.3
    }
    
    # Initialize the model
    model = DecoderOnlyTransformer(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    )
    
    # Load the saved model state_dict
    model_path = 'best_transformer_model.pth'  # Update path if necessary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

@st.cache_resource
def load_tokenizer():
    tokenizer_path = 'urdu_bpe.model'  # Update path if necessary
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp

# ----------------------------
# Generation Functions
# ----------------------------

def generate_text_greedy(model, sp, start_text, sequence_length=50, max_length=100):
    model.eval()
    tokens = sp.encode(start_text, out_type=int)
    generated = tokens.copy()
    
    for _ in range(max_length):
        # Prepare input
        input_seq = torch.tensor([generated[-sequence_length:]], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)  # [1, vocab_size]
            _, next_token = torch.max(output, dim=1)
            next_token = int(next_token.item())  # Ensure pure Python int
        
        generated.append(next_token)
        
        # Stop if EOS token is generated
        if next_token == sp.piece_to_id('</s>'):
            break
    
    return sp.decode_ids(generated)

def generate_text_temperature_sampling(model, sp, start_text, temperature=0.8, sequence_length=50, max_length=100):
    model.eval()
    tokens = sp.encode(start_text, out_type=int)
    generated = tokens.copy()
    
    for _ in range(max_length):
        # Prepare input
        input_seq = torch.tensor([generated[-sequence_length:]], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)  # [1, vocab_size]
            logits = output / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_token = int(next_token)  # Ensure pure Python int
        
        generated.append(next_token)
        
        # Stop if EOS token is generated
        if next_token == sp.piece_to_id('</s>'):
            break
    
    return sp.decode_ids(generated)

def generate_text_top_k(model, sp, start_text, k=50, sequence_length=50, max_length=100):
    model.eval()
    tokens = sp.encode(start_text, out_type=int)
    generated = tokens.copy()
    
    for _ in range(max_length):
        # Prepare input
        input_seq = torch.tensor([generated[-sequence_length:]], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)  # [1, vocab_size]
            probs = torch.softmax(output, dim=-1)
            top_probs, top_indices = torch.topk(probs, k)
            top_probs = top_probs.cpu().numpy().flatten()
            top_indices = top_indices.cpu().numpy().flatten()
            
            # Normalize probabilities
            top_probs = top_probs / top_probs.sum()
            
            # Sample from the top K tokens
            next_token = np.random.choice(top_indices, p=top_probs)
            next_token = int(next_token)  # Ensure pure Python int
        
        generated.append(next_token)
        
        # Stop if EOS token is generated
        if next_token == sp.piece_to_id('</s>'):
            break
    
    return sp.decode_ids(generated)

def generate_text_top_p(model, sp, start_text, p=0.9, sequence_length=50, max_length=100):
    model.eval()
    tokens = sp.encode(start_text, out_type=int)
    generated = tokens.copy()
    
    for _ in range(max_length):
        # Prepare input
        input_seq = torch.tensor([generated[-sequence_length:]], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_seq)  # [1, vocab_size]
            logits = output.squeeze(0)  # [vocab_size]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            # Shift the mask right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_token = int(next_token)  # Ensure pure Python int
        
        generated.append(next_token)
        
        # Stop if EOS token is generated
        if next_token == sp.piece_to_id('</s>'):
            break
    
    return sp.decode_ids(generated)

def generate_text_beam_search(model, sp, start_text, beam_width=5, sequence_length=50, max_length=100):
    model.eval()
    tokens = sp.encode(start_text, out_type=int)
    generated = [[tokens.copy(), 0.0]]  # List of [sequence, score]
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in generated:
            input_seq = torch.tensor([seq[-sequence_length:]], dtype=torch.long)
            with torch.no_grad():
                output = model(input_seq)  # [1, vocab_size]
                probs = torch.softmax(output, dim=-1).squeeze(0)  # [vocab_size]
            
            top_probs, top_indices = torch.topk(probs, beam_width)
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            for prob, idx in zip(top_probs, top_indices):
                candidate_seq = seq + [int(idx)]
                candidate_score = score - math.log(prob + 1e-10)  # Negative log likelihood
                all_candidates.append([candidate_seq, candidate_score])
        
        # Select top beam_width sequences
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        generated = ordered[:beam_width]
        
        # Check if any sequence has generated EOS
        completed = any(seq[-1] == sp.piece_to_id('</s>') for seq, _ in generated)
        if completed:
            break
    
    # Select the sequence with the lowest score
    best_sequence = min(generated, key=lambda tup: tup[1])[0]
    # Ensure all tokens are pure Python integers
    best_sequence = [int(token) for token in best_sequence]
    return sp.decode_ids(best_sequence)

# ----------------------------
# Streamlit Interface
# ----------------------------

def main():
    st.title("✨ Urdu Poetry Generator ✨")
    st.write("Generate beautiful Urdu poetry using a Transformer-based model.")
    
    # Load model and tokenizer
    model = load_model()
    sp = load_tokenizer()
    
    # Sidebar for user inputs
    st.sidebar.header("Input Parameters")
    
    start_text = st.sidebar.text_input("Enter Starting Text:", value="یہ شام", max_chars=100)
    
    generation_method = st.sidebar.selectbox(
        "Select Generation Method:",
        ("Greedy Sampling", "Temperature Sampling", "Top-K Sampling", "Top-P Sampling", "Beam Search")
    )
    
    # Parameters based on generation method
    if generation_method == "Temperature Sampling":
        temperature = st.sidebar.slider("Temperature:", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
    elif generation_method == "Top-K Sampling":
        k = st.sidebar.slider("Top-K Value:", min_value=10, max_value=100, value=50, step=10)
    elif generation_method == "Top-P Sampling":
        p = st.sidebar.slider("Top-P (Nucleus) Value:", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    elif generation_method == "Beam Search":
        beam_width = st.sidebar.slider("Beam Width:", min_value=2, max_value=10, value=5, step=1)
    
    sequence_length = st.sidebar.number_input("Sequence Length:", min_value=10, max_value=100, value=50, step=5)
    max_length = st.sidebar.number_input("Max Generated Tokens:", min_value=50, max_value=500, value=100, step=50)
    
    if st.sidebar.button("Generate Poetry"):
        if not start_text.strip():
            st.sidebar.error("Please enter.")
            return
        
        with st.spinner("Generating poetry"):
            if generation_method == "Greedy Sampling":
                generated_poem = generate_text_greedy(
                    model, sp, start_text, sequence_length=sequence_length, max_length=max_length
                )
            elif generation_method == "Temperature Sampling":
                generated_poem = generate_text_temperature_sampling(
                    model, sp, start_text, temperature=temperature, sequence_length=sequence_length, max_length=max_length
                )
            elif generation_method == "Top-K Sampling":
                generated_poem = generate_text_top_k(
                    model, sp, start_text, k=k, sequence_length=sequence_length, max_length=max_length
                )
            elif generation_method == "Top-P Sampling":
                generated_poem = generate_text_top_p(
                    model, sp, start_text, p=p, sequence_length=sequence_length, max_length=max_length
                )
            elif generation_method == "Beam Search":
                generated_poem = generate_text_beam_search(
                    model, sp, start_text, beam_width=beam_width, sequence_length=sequence_length, max_length=max_length
                )
            else:
                generated_poem = "Invalid generation method selected."
        
        st.subheader("🌟 Generated Poetry 🌟")
        st.write(generated_poem)

if __name__ == "__main__":
    main()

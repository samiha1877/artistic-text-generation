# poetry_generator_app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm
import nltk
from nltk.tokenize import word_tokenize
import os

# Download NLTK data
nltk.download('punkt')

# ---------------------------
# Model Definition
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask: [batch_size, 1, 1, seq_len]
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Reshape and combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.linear_out(output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Self-Attention
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        # Feed Forward
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-Attention
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))
        # Cross-Attention
        cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))
        # Feed Forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, 
                 num_decoder_layers=6, d_ff=2048, dropout=0.1, max_seq_length=100):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # Assuming pad_id=0
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length, dropout=dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        self.fc_out.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc_out.weight)
    
    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        return src_mask.to(torch.bool)
    
    def make_tgt_mask(self, tgt):
        # tgt: [batch_size, tgt_len]
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
        seq_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()  # [seq_len, seq_len]
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        tgt_mask = tgt_pad_mask & subsequent_mask  # [batch_size, 1, seq_len, seq_len]
        return tgt_mask.to(torch.bool)
    
    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Embedding and Positional Encoding
        enc_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))  # [batch, src_len, d_model]
        dec_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))  # [batch, tgt_len, d_model]
        
        # Encoder
        enc_output = enc_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder
        dec_output = dec_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Output
        output = self.fc_out(dec_output)  # [batch, tgt_len, vocab_size]
        return output

# ---------------------------
# Load Tokenizer and Model
# ---------------------------
@st.cache_resource
def load_tokenizer(tokenizer_path='bpe_tokenizer.model'):
    if not os.path.exists(tokenizer_path):
        st.error(f"Tokenizer model file '{tokenizer_path}' not found.")
        return None
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer

@st.cache_resource
def load_model(model_path='best_transformer_model.pth', _tokenizer=None):
    """
    Note: The tokenizer argument is prefixed with an underscore to prevent Streamlit from trying to hash it.
    """
    if _tokenizer is None:
        st.error("Tokenizer must be loaded before loading the model.")
        return None
    vocab_size = _tokenizer.get_piece_size()
    model = TransformerModel(vocab_size=vocab_size).to(device)
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ---------------------------
# Inference Function
# ---------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: logits distribution shape (batch_size, vocab_size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    logits = logits.clone()
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        topk = torch.topk(logits, top_k, dim=-1).values
        logits[logits < topk[:, [-1]]] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def generate_poetry(model, tokenizer, seed_text, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
    """
    Generates a sequence given a source sequence using sampling with temperature, top-k, and top-p filtering.
    
    Args:
        model: Trained Transformer model.
        tokenizer: SentencePieceProcessor instance.
        seed_text: Seed text string to start generation.
        max_length: Maximum length of the generated sequence.
        temperature: Temperature value for scaling logits.
        top_k: Top-k filtering parameter.
        top_p: Top-p (nucleus) filtering parameter.
    
    Returns:
        generated_seq: List of generated token IDs.
    """
    model.eval()
    tokens = tokenizer.encode(seed_text, out_type=int)
    tokens = [tokenizer.bos_id()] + tokens  # Add <s> token
    src = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # [1, src_len]
    
    with torch.no_grad():
        enc_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
        src_mask = model.make_src_mask(src)
        enc_output = enc_emb
        for layer in model.encoder_layers:
            enc_output = layer(enc_output, src_mask)
    
    generated_seq = [tokenizer.bos_id()]
    for _ in range(max_length):
        tgt = torch.tensor(generated_seq, dtype=torch.long).unsqueeze(0).to(device)  # [1, tgt_len]
        tgt_mask = model.make_tgt_mask(tgt)
        
        dec_emb = model.pos_encoder(model.embedding(tgt) * math.sqrt(model.d_model))
        dec_output = dec_emb
        for layer in model.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = model.fc_out(dec_output)  # [1, tgt_len, vocab_size]
        logits = output[:, -1, :] / temperature  # [1, vocab_size]
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)  # [1, vocab_size]
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        generated_seq.append(next_token)
        if next_token == tokenizer.eos_id():
            break
    return generated_seq

# ---------------------------
# Device Configuration
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Poetry Generator", layout="wide")
st.title("✨ Poetry Generator using Transformer Model ✨")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

seed_text = st.sidebar.text_area("Enter Seed Text:", 
                                 "from the forests and the prairies", 
                                 height=100)

temperature = st.sidebar.slider("Temperature:", 0.5, 1.5, 0.8, 0.1)
top_k = st.sidebar.slider("Top-K:", 0, 100, 50, 5)
top_p = st.sidebar.slider("Top-P:", 0.0, 1.0, 0.9, 0.05)
max_length = st.sidebar.slider("Max Length:", 50, 300, 100, 10)

generate_button = st.sidebar.button("Generate Poetry")

# Load tokenizer and model
with st.spinner('Loading tokenizer...'):
    tokenizer = load_tokenizer()
if tokenizer is not None:
    with st.spinner('Loading model...'):
        model = load_model(_tokenizer=tokenizer)  # Note the change here
else:
    st.stop()

# Generate Poetry
if generate_button:
    if seed_text.strip() == "":
        st.warning("Please enter some seed text to generate poetry.")
    else:
        with st.spinner('Generating poetry...'):
            generated_ids = generate_poetry(
                model=model,
                tokenizer=tokenizer,
                seed_text=seed_text,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            # Decode generated tokens
            if tokenizer.eos_id() in generated_ids:
                eos_index = generated_ids.index(tokenizer.eos_id())
                generated_ids = generated_ids[1:eos_index]  # Exclude <s> and </s>
            else:
                generated_ids = generated_ids[1:]
            generated_text = tokenizer.decode(generated_ids)
        
        # Display the generated poem
        st.subheader("🌟 Generated Poem 🌟")
        st.write(generated_text.replace('\n', '\n\n'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from dataloader import get_albhed_dataloader
#original transformer architecture

@dataclass
class AlbhedConfig:
    n_embed: int = 512 # This is C, the embedding dimension of our tokens
    block_size: int = 64 # This is the length of the context window of the sentence i.e T
    n_heads: int = 8 # n_embed must be divisible by n_heads
    n_layers: int = 6 # number of transformer blocks
    dropout: float = 0.1
    head_size: int = n_embed // n_heads # 64
    ff_dim = 4 * n_embed # dimension of feedforward layer
    #vocab_size: int = 50257 # size of the vocabulary, same as GPT-2
    vocab_size: int = 50258 # size of the vocabulary, same as GPT-2 - with custom, 50258 with start
    ignore_pad_token: int = 50257

class Head(nn.Module):
    def __init__(self, config, masked = False):
        super().__init__()
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        self.head_size = config.head_size
        if config.dropout:
            self.dropout = nn.Dropout(config.dropout)
        self.pad_token_id = config.ignore_pad_token if hasattr(config, 'ignore_pad_token') else None

        self.masked = masked

        if self.masked:
            self.register_buffer("tril", torch.tril(torch.ones(self.block_size, self.block_size))) # T, T

        self.Wq = nn.Linear(self.n_embed, self.head_size, bias = False)
        self.Wv = nn.Linear(self.n_embed, self.head_size, bias = False)
        self.Wk = nn.Linear(self.n_embed, self.head_size, bias = False)
        

    def forward(self, x, enc_x = None, pad_mask = None):
        B, T, C = x.size()
        if enc_x is not None:
            k = self.Wk(enc_x)
            v = self.Wv(enc_x)
        else:
            k = self.Wk(x)
            v = self.Wv(x)
        
        q = self.Wq(x)
        
        scores = q @ k.transpose(-2, -1) / C**0.5 # B, T, T
        
        # use masking only for decoder self attention
        if self.masked:
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1) == 0, float('-inf'))
         
        attn = F.softmax(scores, dim=-1) # B, T, T
        
        #apply dropout before multiplying with v
        if hasattr(self, 'dropout'):
            attn = self.dropout(attn)
        
        attn = attn @ v # B, T, head_size
        return attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config, masked = False):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        self.masked = masked
        if config.dropout:
            self.dropout = nn.Dropout(config.dropout)

        self.heads = nn.ModuleList([Head(config, masked=self.masked) for _ in range(self.n_heads)]) # will return B, T, head_size * n_heads
        self.mha_proj = nn.Linear(self.head_size * self.n_heads, config.n_embed) # project to to n_embed B, T, C
        
    def forward(self, x, enc_x = None, pad_mask = None):
        out = torch.cat([h(x, enc_x=enc_x, pad_mask=pad_mask) for h in self.heads], dim = -1) # B, T, head_size * n_heads
        out = self.mha_proj(out)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        return out
    
class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.ff_dim = config.ff_dim
        
        self.in_proj = nn.Linear(self.n_embed, self.ff_dim)
        self.out_proj = nn.Linear(self.ff_dim, self.n_embed)
        self.relu = nn.ReLU()
        
        if config.dropout:
            self.dropout = nn.Dropout(config.dropout)
            
    def forward(self, x):
        x = self.in_proj(x)
        x = self.relu(x)
        x = self.out_proj(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
            
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config, masked=False)
        self.ffn = FFN(config)
        self.norm1 = nn.LayerNorm(config.n_embed)
        self.norm2 = nn.LayerNorm(config.n_embed)
        
    def forward(self, x, pad_mask=None):
        x = x + self.norm1(self.mha(x, pad_mask=pad_mask))
        x = x + self.norm2(self.ffn(x))
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mmha = MultiHeadAttention(config, masked=True)
        self.cross_mha = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.n_embed)
        self.norm2 = nn.LayerNorm(config.n_embed)
        self.norm3 = nn.LayerNorm(config.n_embed)
        self.ffn = FFN(config)
        
    def forward(self, x, enc_x, pad_mask=None, enc_x_pad_mask=None):
        # in cross attention we use the encoder's attention mask not the decoders
        x = x + self.norm1(self.mmha(x, pad_mask=pad_mask))
        x = x + self.norm2(self.cross_mha(x, enc_x=enc_x, pad_mask=enc_x_pad_mask))
        x = x + self.norm3(self.ffn(x))
        
        return x

class Transformer(nn.Module):
    def __init__(self, config: AlbhedConfig):
        super().__init__()
        
        #pad token
        self.ignore_pad_token = config.ignore_pad_token
        
        #token embedding
        self.block_size = config.block_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        
        #position embedding
        self.wpe_x = nn.Embedding(config.block_size, config.n_embed)
        self.wpe_y = nn.Embedding(config.block_size, config.n_embed)
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        
        self.ln_final = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        self.lm_head.weight = self.wte.weight # tie weights with vocab embedding
        
        self.apply(self._init_weights)

    def forward(self, x, y, targets = None, x_pad_mask = None, y_pad_mask = None):
        _, Tx = x.size()
        _, Ty = y.size()

        assert Tx < self.block_size, "Source sentence length exceeds block size"
        assert Ty < self.block_size, "Target sentence length exceeds block size"

        pos_x = torch.arange(0, Tx, dtype=torch.long, device=x.device)
        pos_y = torch.arange(0, Ty, dtype=torch.long, device=y.device)
        
        x_tok_emb = self.wte(x) # B, T, C
        x_pos_emb = self.wpe_x(pos_x) # T, C
        
        x = x_tok_emb + x_pos_emb #broadcasting used
        
        y_tok_emb = self.wte(y) # B, T, C
        y_pos_emb = self.wpe_y(pos_y) # T, C

        y = y_tok_emb + y_pos_emb
        
        for block in self.encoder_blocks:
            x = block(x, pad_mask=x_pad_mask)

        for block in self.decoder_blocks:
            y = block(y, enc_x=x, pad_mask=y_pad_mask, enc_x_pad_mask=x_pad_mask)

        y = self.ln_final(y)
        logits = self.lm_head(y) # B, T, vocab_size
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.ignore_pad_token)

        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


# -------------- Running Code --------------

print("--- Transformer Model Loading ---")
model = Transformer(AlbhedConfig())
print("Model loaded successfully!")


dataloader = get_albhed_dataloader(batch_size=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(250):
    data = next(iter(dataloader))
    encoder_input_ids = data['encoder_input_ids']
    decoder_input_ids = data['decoder_input_ids']
    labels = data['labels']
    encoder_attention_mask = data['encoder_attention_mask']
    decoder_attention_mask = data['decoder_attention_mask']

    optimizer.zero_grad()

    logits, loss = model(
        x = encoder_input_ids, 
        y = decoder_input_ids, 
        targets=labels, 
        x_pad_mask=encoder_attention_mask, 
        y_pad_mask=decoder_attention_mask
    )

    loss.backward()
    optimizer.step()
    
    print(f"step {i+1}, loss: {loss.item()}")
import os
import math
import time
import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass

# Initialize wandb
wandb.init(project="shakespeare-gpt", name="gpt2-124M-training")

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                  .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Print input parameters and model architecture
def print_model_params(model, config):
    """Print model parameters and configuration"""
    print("\n=== Model Configuration ===")
    print(f"Model Architecture: GPT-2 (124M parameters)")
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of heads: {config.n_head}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Block size (context length): {config.block_size}")
    print(f"Vocabulary size: {config.vocab_size}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def print_training_params(batch_size, block_size, learning_rate, max_iters, device):
    """Print training hyperparameters"""
    print("\n=== Training Parameters ===")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Maximum iterations: {max_iters}")
    print(f"Training device: {device}")
    print(f"Evaluation interval: {eval_interval}")
    print(f"Evaluation iterations: {eval_iters}")

# Training settings - Updated device configuration for Mac M2
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"\n=== Device Configuration ===")
print(f"Using device: {device}")
print(f"MPS (Mac GPU) available: {torch.backends.mps.is_available()}")
print(f"CUDA (NVIDIA GPU) available: {torch.cuda.is_available()}")

# Update autocast for M2 compatibility
def get_autocast_device():
    if device.type == 'mps':
        return 'cpu'  # MPS doesn't support autocast yet, fallback to CPU
    return device.type

# Updated Training settings with longer sequences and better hyperparameters
batch_size = 16  # Balanced for longer sequences
block_size = 1024  # Increased to 1024 for more context
learning_rate = 6e-4
max_iters = 10000
eval_interval = 100
eval_iters = 20
grad_clip = 1.0
weight_decay = 0.1

# Learning rate scheduler parameters - adjusted for longer training
warmup_iters = 2000  # Increased warmup for stability
lr_decay_iters = 8000
min_lr = 6e-5

# Add gradient accumulation for larger effective batch size
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps

print(f"\n=== Training Configuration ===")
print(f"Sequence length (block_size): {block_size}")
print(f"Batch size: {batch_size}")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Effective batch size: {effective_batch_size}")

# Add learning rate scheduler function
def get_lr(it):
    # Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine learning rate decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Load and encode text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
data = torch.tensor(enc.encode(text), dtype=torch.long)

# Print dataset statistics
print("\n=== Dataset Statistics ===")
print(f"Total text length: {len(text):,} characters")
print(f"Total tokens: {len(data):,}")
print(f"Unique tokens: {len(set(data.tolist())):,}")

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Training set size: {len(train_data):,} tokens")
print(f"Validation set size: {len(val_data):,} tokens")

# Initialize model and print parameters
model = GPT(GPTConfig())
print_model_params(model, model.config)
print_training_params(batch_size, block_size, learning_rate, max_iters, device)

model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=weight_decay
)

print("\n=== Starting Training ===")

# Modified training loop with gradient accumulation
best_val_loss = float('inf')
training_start_time = time.time()

# Set initial gradients to zero
optimizer.zero_grad(set_to_none=True)

for iter in range(max_iters):
    # Get the current learning rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_interval == 0:
        losses = []
        model.eval()
        for k in range(eval_iters):
            X, Y = get_batch(val_data, batch_size, block_size, device)
            with torch.no_grad():
                with torch.autocast(device_type=get_autocast_device()):
                    logits, loss = model(X, Y)
            losses.append(loss.item())
        val_loss = torch.tensor(losses).mean()
        
        wandb.log({
            "iter": iter,
            "val_loss": val_loss,
            "learning_rate": lr
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
                'val_loss': val_loss,
            }, 'best_model.pt')
            
        print(f"step {iter}: val loss {val_loss:.4f} | best loss {best_val_loss:.4f} | lr {lr:.2e}")
        
        if val_loss < 0.099999:
            training_time = time.time() - training_start_time
            print(f"\nTarget loss achieved at iteration {iter}")
            print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
            break
        
        model.train()

    # Gradient accumulation loop
    accumulated_loss = 0
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch(train_data, batch_size, block_size, device)
        with torch.autocast(device_type=get_autocast_device()):
            logits, loss = model(X, Y)
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps
        
        # Scale the loss to prevent underflow
        if device.type == 'mps':
            scaled_loss = loss * batch_size
        else:
            scaled_loss = loss
            
        scaled_loss.backward()
        accumulated_loss += loss.item()

    # Clip gradients and update weights after accumulation
    if (iter + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        wandb.log({
            "iter": iter,
            "train_loss": accumulated_loss,
            "learning_rate": lr
        })

# Modified generation settings for better output
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40, device=device):
    """Generate text with improved sampling"""
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # Forward pass
        with torch.autocast(device_type=get_autocast_device()):
            logits, _ = model(idx_cond)
        # Focus on last time step
        logits = logits[:, -1, :] / temperature
        # Crop logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # Apply softmax to convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Generate sample text with improved settings
print("\n=== Generating Sample Text ===")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nGenerated text:")
generated = generate(model, context, max_new_tokens=1000)  # Generate longer sample
print(enc.decode(generated[0].tolist()))

wandb.finish() 
import os
# Set MPS memory management at the very beginning of the file
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Remove memory limit
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'   # Aggressive memory cleanup

# Updated Training settings with more aggressive memory optimization
batch_size = 4  # Further reduced from 8
block_size = 256  # Further reduced from 512
learning_rate = 6e-4
max_iters = 15000  # Increased to compensate for smaller batch
eval_interval = 100
eval_iters = 5  # Further reduced from 10
grad_clip = 1.0
weight_decay = 0.1

# Gradient accumulation steps increased
gradient_accumulation_steps = 16  # Increased from 8
effective_batch_size = batch_size * gradient_accumulation_steps  # Still 64

# Memory management function
def clear_memory():
    if device.type == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    if 'memory_allocated' in dir(torch.mps):
        print(f"Memory allocated: {torch.mps.memory_allocated() / 1024**2:.2f} MB")

# Modified training loop with more frequent memory clearing
for iter in range(max_iters):
    # Clear memory every 50 iterations
    if iter % 50 == 0:
        clear_memory()
        print_memory_usage()
    
    # Gradient accumulation loop
    accumulated_loss = 0
    optimizer.zero_grad(set_to_none=True)
    
    for micro_step in range(gradient_accumulation_steps):
        # Get batch and free memory
        X, Y = get_batch(train_data, batch_size, block_size, device)
        
        with torch.autocast(device_type=get_autocast_device()):
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
        # Scale loss and backward pass
        scaled_loss = loss * batch_size if device.type == 'mps' else loss
        scaled_loss.backward()
        accumulated_loss += loss.item()
        
        # Free memory after backward pass
        del X, Y, logits, loss, scaled_loss
        if micro_step % 4 == 0:  # Clear memory every 4 micro steps
            clear_memory()
    
    # Clip gradients and update weights
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    clear_memory()  # Clear memory after optimizer step

import psutil

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Add memory monitoring to training loop
if iter % eval_interval == 0:
    print_memory_usage()

@dataclass
class GPTConfig:
    block_size: int = 512  # Reduced from 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40, device=device):
    """Generate text with improved sampling and memory efficiency"""
    model.eval()
    
    # Clear cache before generation if using MPS
    if device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    for _ in range(max_new_tokens):
        # Process in smaller chunks if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        with torch.autocast(device_type=get_autocast_device()):
            logits, _ = model(idx_cond)
            
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Periodically clear cache during generation
        if _ % 100 == 0 and device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            
    return idx 
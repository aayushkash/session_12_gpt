import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from huggingface_hub import login

# Authenticate with Hugging Face
def init_huggingface():
    try:
        # Get token from environment variable or use default
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            login(hf_token)
            print("Successfully logged in to Hugging Face")
        else:
            print("No Hugging Face token found, trying anonymous access")
    except Exception as e:
        print(f"Authentication error: {e}")

# Load model and tokenizer
def load_model():
    try:
        # Initialize Hugging Face authentication
        init_huggingface()
        
        print("Loading model...")
        # Try loading with auth token first
        model = GPT2LMHeadModel.from_pretrained(
            "aayushraina/gpt2shakespeare",
            local_files_only=False,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        print("Loading tokenizer...")
        # Use the base GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print("Tokenizer loaded successfully!")
        
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        try:
            # Fallback to base GPT-2 if custom model fails
            print("Attempting to load base GPT-2 model as fallback...")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            print("Fallback successful - loaded base GPT-2")
            return model, tokenizer
        except Exception as e:
            print(f"Fallback failed: {e}")
            return None, None

# Text generation function
def generate_text(prompt, max_length=500, temperature=0.8, top_k=40, top_p=0.9):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Load model and tokenizer globally
print("Loading model and tokenizer...")
model, tokenizer = load_model()
print("Model loaded successfully!")

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt", placeholder="Start your text here...", lines=2),
        gr.Slider(minimum=10, maximum=1000, value=500, step=10, label="Maximum Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-k"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Shakespeare-style Text Generator",
    description="""Generate Shakespeare-style text using a fine-tuned GPT-2 model.
    
    Parameters:
    - Temperature: Higher values make the output more random, lower values more focused
    - Top-k: Number of highest probability vocabulary tokens to keep for top-k filtering
    - Top-p: Cumulative probability for nucleus sampling
    """,
    examples=[
        ["First Citizen:", 500, 0.8, 40, 0.9],
        ["To be, or not to be,", 500, 0.8, 40, 0.9],
        ["Friends, Romans, countrymen,", 500, 0.8, 40, 0.9],
        ["O Romeo, Romeo,", 500, 0.8, 40, 0.9],
        ["Now is the winter of our discontent", 500, 0.8, 40, 0.9]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch() 
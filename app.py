import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer from Hugging Face
def load_model():
    model_name = "aayushraina/gpt2shakespeare"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

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
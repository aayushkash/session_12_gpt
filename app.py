import torch
import gradio as gr
import tiktoken
from train_shakespeare import GPT, GPTConfig, generate, get_autocast_device

# Load the trained model
def load_model():
    model = GPT(GPTConfig())
    checkpoint = torch.load('best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Text generation function
def generate_text(prompt, max_length=500, temperature=0.8, top_k=40):
    # Tokenize the prompt
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    
    # Generate text
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode the generated text
    generated_text = enc.decode(output_ids[0].tolist())
    return generated_text

# Load model globally
model = load_model()

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=1000, value=500, step=10, label="Maximum Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-k")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Shakespeare-style Text Generator",
    description="Generate Shakespeare-style text using a fine-tuned GPT-2 model",
    examples=[
        ["First Citizen:", 500, 0.8, 40],
        ["To be, or not to be,", 500, 0.8, 40],
        ["Friends, Romans, countrymen,", 500, 0.8, 40]
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch() 
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Load the fine-tuned model and tokenizer
model_path = r"D:\human\fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, model, tokenizer, max_length=150):
    # Encode the input text
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response using the model
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated tokens into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    
    return response

# Streamlit interface
st.title("Chatbot")

user_input = st.text_input("You: ")

if user_input:
    prompt = f"User: {user_input} Bot:"
    response = generate_response(prompt, model, tokenizer)
    st.write(f"Bot: {response}")

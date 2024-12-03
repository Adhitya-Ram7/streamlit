# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load the base model, tokenizer, and PEFT model
@st.cache_resource
def load_model():
    # PEFT model and base model IDs
    peft_model_id = "Aryaman02/legalpara-lm"
    base_model_id = "allenai/Llama-3.1-Tulu-3-8B"
    
    # Configure quantization
    bnb_config = None  # Disable bitsandbytes configuration
    
    # Load the base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="cpu",
        trust_remote_code=True,
        token="hf_lMgiyTyBYrwYsBMctUEgjZIQeQenfCinHEk"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token="hf_lMgiyTyBYrwYsBMctUEgjZIQeQenfCinHEk"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create PEFT model configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)

    return model, tokenizer, peft_model

# Helper function to generate legal responses
def generate_legal_response(prompt, model, tokenizer, peft_model, max_new_tokens=4096):

    # Define the system prompt
    SYSTEM_PROMPT = """You are an experienced Indian lawyer and legal expert, equipped with comprehensive knowledge of Indian law, including the latest amendments, judgments, and legal precedents up to 2024. You have:

1. Deep expertise in Indian constitutional law, civil law, criminal law, and specialized areas like corporate law, intellectual property, and family law
2. Thorough understanding of the Indian judicial system, from lower courts to the Supreme Court
3. Knowledge of recent legislative changes and landmark judgments
4. Familiarity with both theoretical and practical aspects of Indian legal practice
5. Expertise in legal documentation, interpretation, and procedure

Please provide accurate, well-reasoned legal advice while citing relevant sections, cases, and precedents when applicable. Always maintain professional ethics and clearly state when a matter requires in-person legal consultation."""
    
    # Combine system prompt with the user's question
    full_prompt = f"{SYSTEM_PROMPT}\n\nLegal Question: {prompt}\n\nAnswer:"

    # Tokenize the input and generate a response
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(peft_model.device)
    outputs = peft_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.95,
        repetition_penalty=1.15
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# Streamlit app starts here
st.set_page_config(
    page_title="LawMate AI Legal Advisor",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ LawMate AI Legal Advisor")
st.write("Welcome to **LawMate**, your AI-powered legal assistant specializing in Indian law. Ask your legal questions below!")

# Sidebar for instructions and credits
with st.sidebar:
    st.header("About LawMate")
    st.write("""
    LawMate is an advanced AI tool designed to assist with legal queries related to Indian law.
    It provides accurate, well-reasoned legal advice while citing relevant sections and precedents.
    """)
    st.write("Developed by Adhityanarayan Ramkumar © 2024")
    st.write("### Usage Instructions:")
    st.write("""
    1. Enter your legal question in the text box.
    2. Click 'Get Response' to see the AI-generated advice.
    3. Note: This tool is for informational purposes only and not a substitute for professional legal advice.
    """)

# Input area
question = st.text_area("Enter your legal question:", height=200)
generate_button = st.button("Get Response")

# Generate response on button click
if generate_button:
    if not question.strip():
        st.error("Please enter a legal question before clicking 'Get Response'.")
    else:
        # Load the model and tokenizer
        with st.spinner("Generating response... Please wait."):
            model, tokenizer, peft_model = load_model()
            response = generate_legal_response(question, model, tokenizer, peft_model)

        # Display the response
        st.subheader("AI Response:")
        st.write(response)

        # Option to download the response as a text file
        st.download_button(
            label="Download Response as Text",
            data=response,
            file_name="legal_response.txt",
            mime="text/plain"
        )

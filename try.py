import os
import subprocess

# Install packages if not already installed
packages = ['transformers', 'torch', 'peft', 'fpdf']
for package in packages:
    try:
        __import__(package.split('==')[0])  # Check if package is already installed
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])

# Import required packages
import streamlit as st
from transformers import AutoTokenizer
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from fpdf import FPDF

# Load the base model, tokenizer, and PEFT model
@st.cache_resource
def load_model():
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

    full_prompt = f"{SYSTEM_PROMPT}\n\nLegal Question: {prompt}\n\nAnswer:"
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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# Helper function to generate PDF
def generate_pdf(response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, response)
    return pdf.output(dest='S').encode('latin1')

# Get dynamic disclaimers
def get_disclaimer(category):
    disclaimers = {
        "Criminal Law": "This advice does not substitute professional legal counsel in criminal matters.",
        "Family Law": "For sensitive matters like divorce, personal consultation is recommended.",
        "Corporate Law": "Corporate legal advice requires in-depth knowledge of your business context."
    }
    return disclaimers.get(category, "This is general legal advice. Consult a professional for case-specific guidance.")

# Streamlit app starts here
st.set_page_config(
    page_title="LawMate AI Legal Advisor",
    page_icon="⚖️",
    layout="centered"
)

# Custom CSS to center the logo and adjust position
st.markdown("""
    <style>
        .stApp {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: center;
            height: 100vh;
        }
        .stImage>img {
            margin-top: 20px;  /* Push the logo towards the top */
            max-width: 100%;   /* Make the logo responsive */
        }
    </style>
""", unsafe_allow_html=True)

# Adding logo and centering it
st.image("logo.jpg", width=150)
st.title("⚖️ LawMate AI Legal Advisor")
st.write("Welcome to **LawMate**, your AI-powered legal assistant specializing in Indian law. Ask your legal questions below!")

# Sidebar with features
with st.sidebar:
    st.header("LawMate Features")
    
    # Conversation history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for i, (q, r) in enumerate(st.session_state.chat_history):
            st.write(f"Q{i+1}: {q}")
            st.write(f"A{i+1}: {r}")

    # About us
    st.subheader("About Us")
    st.write("**LawMate AI** was developed to assist users with Indian legal queries. This project is maintained by Adhityanarayan Ramkumar.")

# Inject custom CSS for text area styling (no 'style' attribute passed to `st.text_area()`)
st.markdown("""
    <style>
        .stTextArea>div>div>textarea {
            background-color: #F0E0B6;  /* Light Beige color */
            border: 1px solid #D9D9D9;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Input area
question = st.text_area(
    "Enter your legal question:", 
    height=200, 
    placeholder="Type your question here...",
    key="question_input",
    label_visibility="collapsed",
    help="Please enter a detailed legal question.",
    max_chars=500,
    disabled=False
)
generate_button = st.button("Get Response")

# Generate response on button click
if generate_button:
    if not question.strip():
        st.error("Please enter a valid legal question before clicking 'Get Response'.")
    elif len(question.strip()) < 10:
        st.warning("Your question seems too short. Please provide more details.")
    else:
        # Load the model and tokenizer
        with st.spinner("Generating response... Please wait."):
            model, tokenizer, peft_model = load_model()
            response = generate_legal_response(question, model, tokenizer, peft_model)
        
        st.session_state.chat_history.append((question, response))
        st.subheader("AI Response:")
        st.write(response)

        # Add a dynamic disclaimer
        category = "General"  # Replace with detected category logic if needed
        st.info(get_disclaimer(category))
        
        # PDF download button
        pdf_data = generate_pdf(response)
        st.download_button(
            label="Download Response as PDF",
            data=pdf_data,
            file_name="legal_response.pdf",
            mime="application/pdf"
        )

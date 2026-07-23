import google.generativeai as genai
import streamlit as st

# Pulls the key securely from your .streamlit/secrets.toml
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

print("\n--- YOUR ACTIVE GEMINI MODELS ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
print("---------------------------------\n")
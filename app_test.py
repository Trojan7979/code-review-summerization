# requirements.txt
# streamlit==1.33.0
# transformers==4.40.1
# torch==2.2.1
# langchain==0.1.16
# huggingface_hub==0.23.0
# librosa==0.10.1
# python-dotenv==1.0.0
# soundfile==0.12.1

import streamlit as st
import tempfile
import os
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login
import torch
import librosa
import numpy as np
import soundfile as sf

def authenticate_hf():
    if 'HF_KEY' in st.secrets:
        hf_token = st.secrets.HF_KEY
    else:
        hf_token = st.text_input("Enter your Hugging Face token:", type="password")
    
    if hf_token:
        try:
            login(token=hf_token)
            return hf_token
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return None
    return None

@st.cache_resource
def load_models(hf_token):
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if torch.cuda.is_available() else -1,
        token=hf_token
    )

    summarization_pipeline = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1,
        token=hf_token
    )

    hf_llm = HuggingFacePipeline(pipeline=summarization_pipeline)

    prompt_template = """Summarize the following content concisely:
    
    {text}
    
    CONCISE SUMMARY:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"]
    )

    summarization_chain = LLMChain(
        llm=hf_llm,
        prompt=prompt,
        verbose=False
    )

    return asr_pipeline, summarization_chain

def transcribe_audio(audio_path, asr_pipeline):
    y, sr = librosa.load(audio_path, sr=16000)
    audio_array = np.array(y, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio_array, sr)
        result = asr_pipeline(temp_file.name)
        os.unlink(temp_file.name)
    
    return result["text"]

def generate_report(transcript, summary):
    return f"""
    Video Analysis Report
    =====================
    
    **Transcript:**
    {transcript}
    
    **Summary:**
    {summary}
    
    **Statistics:**
    - Total Words: {len(transcript.split())}
    - Summary Ratio: {len(summary.split())/len(transcript.split()):.1%}
    """

def main():
    st.title("🎥 Video to Report Generator")
    st.markdown("Upload an audio file (WAV or MP3) to extract text and generate summary")

    hf_token = authenticate_hf()
    if not hf_token:
        st.stop()

    asr_pipeline, summarization_chain = load_models(hf_token)

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name
            
            transcript = transcribe_audio(audio_path, asr_pipeline)
            summary = summarization_chain.run(transcript)
            
            os.unlink(audio_path)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Transcript")
                st.text_area("", transcript, height=300)
            
            with col2:
                st.subheader("Summary")
                st.write(summary)
            
            st.download_button(
                "📥 Download Report",
                generate_report(transcript, summary),
                file_name="report.txt"
            )

if __name__ == "__main__":
    main()
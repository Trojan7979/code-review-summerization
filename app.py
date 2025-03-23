import streamlit as st
import os
import tempfile
import moviepy.editor as mp
import speech_recognition as sr
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from github_integration import get_github_content

# Set page configuration
st.set_page_config(page_title="AI-Powered Code Review Summerization from Meeting Recordings", layout="wide")

# App title and description
st.title("AI-Powered Code Review Summerization from Meeting Recordings")
st.write("Upload a video to extract and summarize spoken content and generate a report")

# Initialize session state variables if they don't exist
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# Function to extract audio from video
def extract_audio(video_file):
    # Create temporary files with auto-deletion turned off
    temp_video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    # Save paths
    video_path = temp_video_file.name
    audio_path = temp_audio_file.name
    
    # Close file handles immediately
    temp_video_file.close()
    temp_audio_file.close()
    
    # Write video data to the temp file
    with open(video_path, 'wb') as f:
        f.write(video_file.read())
    
    # Extract audio from video
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        # Close video file before trying to delete it
        video.close()
        
        # Try to delete the video file, but don't raise an error if it fails
        try:
            os.unlink(video_path)
        except (PermissionError, OSError):
            # We'll leave cleanup to the OS later if we can't delete now
            pass
        
        return audio_path
    except Exception as e:
        # Make sure to close the video in case of any errors
        try:
            video.close()
        except:
            pass
        raise e

# Function to transcribe audio to text
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    # Split audio into chunks to handle longer files
    audio = sr.AudioFile(audio_path)
    full_text = ""
    
    # Get audio duration before opening the file for processing
    audio_duration = mp.AudioFileClip(audio_path).duration
    audio_clip = mp.AudioFileClip(audio_path)
    
    with audio as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        # Process audio in chunks to handle longer files
        chunk_duration = 30  # seconds
        offset = 0
        
        while offset < audio_duration:
            audio_data = recognizer.record(source, duration=min(chunk_duration, audio_duration - offset))
            try:
                text = recognizer.recognize_google(audio_data)
                full_text += text + " "
            except sr.UnknownValueError:
                full_text += "[inaudible] "
            except sr.RequestError as e:
                return f"Error: {e}"
            
            offset += chunk_duration
    
    # Try to delete the audio file, but don't raise an error if it fails
    try:
        # Close any AudioFileClip that might be open
        audio_clip.close()
        os.unlink(audio_path)
    except (PermissionError, OSError):
        # We'll leave cleanup to the OS later if we can't delete now
        pass
    
    return full_text.strip()

# Direct summarization function
def summarize_text(text, model_name, huggingface_api_token):
    # Set the correct environment variable for Hugging Face
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    
    # Initializing the language model
    llm = HuggingFaceHub(
        repo_id=model_name,

        huggingfacehub_api_token=huggingface_api_token,
        model_kwargs={
            "temperature": 0.3,
            "max_length": 1024,
            "max_new_tokens": 512
        }
    )
    
    # Creates a summarization prompt
    summarize_prompt_template = """
    Summarize the following text in a concise and informative way:
    
    {text}
    
    Summary:
    """
    
    # Creates LLM chain for summarization
    summarize_prompt = PromptTemplate(template=summarize_prompt_template, input_variables=["text"])
    summary_chain = LLMChain(llm=llm, prompt=summarize_prompt)
    
    # Generating summary
    result = summary_chain.run(text=text)
    return result

# Direct summarization function
def report_generation(text, model_name, huggingface_api_token, github_url, github_pat):
    # Set the correct environment variable for Hugging Face
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    
    # Initializing the language model
    llm = HuggingFaceHub(
        repo_id=model_name,
        huggingfacehub_api_token=huggingface_api_token,
        model_kwargs={
            "temperature": 0.3,
            "max_length": 1024,
            "max_new_tokens": 512
        }
    )
    
    res, contributors_response = get_github_content(github_url, github_pat)
    
   # Defining the prompt template with a clear separator
    report_generation_prompt_template = """
    You are an expert code reviewer. Your task is to generate a very detailed report based on the Meeting discussion and Github code provided.
    The report should be well-structured and should highlight areas where the code needs improvement or changes.
    Also assign the tasks to developers who are working on their repository and who will be liable to make the changes as following, after the code review and check the meeting discussion if there is any developer being mentioned.
    If there exist no such mention, then assign the task to the GitHub contributor. Make this task section after this report. Generate the report in markdown format.

    Meeting Discussion:
    {text}

    Github Code Summary:
    {res}

    REPORT:
    """

    # Create prompt template
    report_generation = PromptTemplate(template=report_generation_prompt_template, input_variables=["text", "res", "contributors_response"])

    # Create LLM chain with return_only_outputs=True
    report_generation_chain = LLMChain(
        llm=llm, 
        prompt=report_generation
    )

    # Generate report - this will return only the output
    raw_result = report_generation_chain.run(text=text, res=res, contributors_response=contributors_response)
    if "REPORT:" in raw_result:
        result = raw_result.split("REPORT:")[1].strip()
    else:
        result = raw_result
    return result

# Sidebar for API key and model inputs
with st.sidebar:
    st.header("Configuration")
    huggingface_api_token = st.text_input("Enter Hugging Face API Token", type="password")
    model_names = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-3.2-3B-Instruct"] 
    model_name = st.selectbox("Enter Model Name (e.g., mistralai/Mistral-7B-Instruct-v0.2)", model_names)
    
    github_url = st.text_input("Enter your github url")
    github_pat = st.text_input("Enter your personal access token")
    
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Enter your Hugging Face API token")
    st.markdown("2. Enter the model name to use for summarization")
    st.markdown("3. Upload a video file")
    st.markdown("4. Click 'Process Video' to extract text")
    st.markdown("5. The extracted text and summary will appear below")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])

# Process button
if uploaded_file is not None and st.button("Process Video"):
    if not huggingface_api_token:
        st.error("Please enter your Hugging Face API token in the sidebar")
    elif not model_name:
        st.error("Please enter a model name in the sidebar")
    elif not github_url:
        st.error("Please enter a Github url in the sidebar")
    elif not github_pat:
        st.error("Please enter a Github Personal Access token in the sidebar")
    
    else:
        with st.spinner("Processing video..."):
            try:
                # Extract audio from video
                audio_path = extract_audio(uploaded_file)
                
                # Transcribe audio to text
                extracted_text = speech_to_text(audio_path)
                st.session_state.extracted_text = extracted_text
                
                if extracted_text and not extracted_text.startswith("Error:"):
                    # Summarize text
                    with st.spinner(f"Summarizing content using {model_name}..."):
                        try:
                            summary = summarize_text(extracted_text, model_name, huggingface_api_token)
                            res = report_generation(summary, model_name, huggingface_api_token, github_url, github_pat)
                            # prompt = """
                            #     This is the summary of this video : {summary}
                            #     """
                            st.session_state.summary = res
                        except Exception as e:
                            st.error(f"Error in summarization: {str(e)}")
                            st.error("Please verify your API token and model name are correct.")
                else:
                    st.error(f"Error in text extraction: {extracted_text}")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

# Display results in tabs
if st.session_state.extracted_text or st.session_state.summary:
    tab1, tab2 = st.tabs(["Extracted Text", "Summary"])
    
    with tab1:
        st.subheader("Extracted Text")
        st.text_area("Full Transcript", st.session_state.extracted_text, height=300)
        
        # Add download button for text
        if st.session_state.extracted_text:
            st.download_button(
                label="Download Transcript",
                data=st.session_state.extracted_text,
                file_name="transcript.txt",
                mime="text/plain"
            )
    
    with tab2:
        st.subheader("Summary")
        st.markdown(st.session_state.summary)
        
        # Add download button for summary
        if st.session_state.summary:
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name="summary.txt",
                mime="text/plain"
            )
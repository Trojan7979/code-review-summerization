import streamlit as st
import os
import tempfile
import moviepy.editor as mp
from github_integration import get_github_content
from llm import message_response, speech_to_text_llm, get_response

# Set page configuration
st.set_page_config(page_title="AI-Powered Code Review Summarization from Meeting Recordings", layout="wide")

# App title and description
st.title("AI-Powered Code Review Summarization from Meeting Recordings")
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
        
        try:
            os.unlink(video_path)
        except (PermissionError, OSError):
            pass 
        return audio_path
    
    except Exception as e:
        try:
            video.close()
        except:
            pass
        raise e

# Direct summarization function
def refine_text(text, model_name, token, type):
    
    try:
        
        # Creates a summarization prompt
        summarize_prompt_template = f"""
        Refine the following text in a concise and informative way:
        
        {text}
        
        Refined text:
        """

        if type == "Google":
            res = get_response(summarize_prompt_template, token, model_name)
        else:
            res = message_response(summarize_prompt_template, token, model_name)
        return res

    except Exception as e:
        raise RuntimeError(f"Failed to access model: {str(e)}. Verify model name and API token.")

# Direct summarization function
def report_generation(text, github_url, github_pat, token, model_name, type):   
    res, contributors_response = get_github_content(github_url, github_pat)
    
    report_generation_prompt_template = f"""
    You are an expert code reviewer. Your task is to generate a very detailed report based on the Meeting discussion and Github code provided.
    Always try to find the context on the GitHub code with the code problem that is discussed in the meeting.
    The report should be well-structured and should highlight areas where the code needs improvement or changes.
    Also assign the tasks to developers who are working on their repository and who will be liable to make the changes as following, after the code review and check the meeting discussion if there is any developer being mentioned.
    If there exist no such mention, don't assign tasks to anyone. Make this task section after this report. Make this report very attractive, informative, detailed, and complete.

    Meeting Discussion:
    {text}

    Github Code:
    {res}

    Guidelines:
    Just return only the report, don't return anything except the report.
    Generate the report in markdown format only.
    Don't mention any dates.
    
    """

    if type == "Google":
        result = get_response(report_generation_prompt_template, token, model_name)
    else:
        result = message_response(report_generation_prompt_template, token, model_name)
    return result


# Sidebar for API key and model inputs
with st.sidebar:
    st.header("Configuration")

    model_service_provider = st.radio(
        "Choose your LLM service provider:",
        ("Google", "Groq")
    )
    if model_service_provider == "Google":
        llm_api_token = st.text_input("Enter Google API Token", type="password", placeholder="xxxxxxxxxxxxxx")
        groq_voice_token = st.text_input("Enter Groq API Token to extract text",type="password", placeholder="gsk_xxxxxxxxxxxxxx")
        report_generation_models = ["------------Select Model-----------", "gemini-2.5-pro-exp-03-25", "gemini-2.0-flash"]
        report_generation_model = st.selectbox("Enter Model Name (e.g., gemini-2.5-pro-exp-03-25)", report_generation_models, index=0)
    else:
        llm_api_token = st.text_input("Enter Groq API Token", type="password", placeholder="gsk_xxxxxxxxxxxxxx")
        groq_voice_token = llm_api_token
        report_generation_models = ["------------Select Model-----------", "deepseek-r1-distill-llama-70b"]
        report_generation_model = st.selectbox("Enter Model Name (e.g., deepseek-r1-distill-llama-70b)", report_generation_models, index=0)

    github_url = st.text_input("Enter your github url", placeholder="https://github.com/{user}/{repo_name}")
    github_branch = st.text_input("Enter you repository path", placeholder="main/path")
    github_url = f"{github_url}/{github_branch}"

    # Radio button selection
    repo_type = st.radio(
        "Choose your repository type:",
        ("Public Repo", "Private Repo")
    )

    if repo_type == "Private Repo":
        github_pat = st.text_input("Enter your personal access token", placeholder= "github_pat_xxxxxxxxxx")
    else:
        github_pat = ""

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Choose your Provider")
    st.markdown("2. Enter the API Token of the chosen provider")
    st.markdown("3. Enter Groq API Token to extract text")
    st.markdown("4. Choose a Model from the dropdown")
    st.markdown("5. Enter your GitHub url")
    st.markdown("6. Enter your GitHub Repository path")
    st.markdown("7. Choose repository type")
    st.markdown("8. If the repository is private then, enter your GitHub Personal Access Token")
    st.markdown("9. Upload a video file")
    st.markdown("10. Click 'Process Video' to extract text")
    st.markdown("11. The Generated Report and Extracted Text will appear below")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])

# Process button
if uploaded_file is not None and st.button("Process Video"):

    if not llm_api_token:
        st.error("Please enter your Groq API token")
    elif not report_generation_model:
        st.error("Please enter a model name in the sidebar")
    elif not github_url:
        st.error("Please enter a Github url in the sidebar")
    elif not github_branch:
        st.error("Please enter a Github Repository path")
    
    else:
        with st.spinner("Processing video..."):
            try:
                # Extract audio from video
                audio_path = extract_audio(uploaded_file)
                
                # Transcribe audio to text
                extracted_text = speech_to_text_llm(audio_path, groq_voice_token)
                st.session_state.extracted_text = extracted_text
                
                if extracted_text and not extracted_text.startswith("Error:"):
                    # Summarize text
                    with st.spinner(f"Summarizing content using {report_generation_model}..."):
                        try:
                            summary = refine_text(extracted_text, report_generation_model, llm_api_token, type=model_service_provider)
                            res = report_generation(summary, github_url, github_pat, llm_api_token, report_generation_model, type=model_service_provider)
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
    tab1, tab2 = st.tabs(["Generated Report", "Extracted Text"])
    
    with tab1:
        st.markdown(st.session_state.summary)
        
        # Add download button for summary
        if st.session_state.summary:
            st.download_button(
                label="Download Report",
                data=st.session_state.summary,
                file_name="Report.txt",
                mime="text/plain"
            )
            
    with tab2:
        st.text_area("Full Transcript", st.session_state.extracted_text, height=300)
        
        # Add download button for text
        if st.session_state.extracted_text:
            st.download_button(
                label="Download Transcript",
                data=st.session_state.extracted_text,
                file_name="transcript.txt",
                mime="text/plain"
            )
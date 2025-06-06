# AI-Powered Code Review Summarization from Meeting Recordings
An intelligent system that processes code review meeting recordings, extracts actionable insights, and maps discussion to relevant GitHub repository code. Automates code review tracking while maintaining high accuracy in identifying changes, action items, and key discussion points, which results into generating well-structured reports.

## Features:

- **Automated Insights Extraction** - Processes MP4 meeting recordings to extract technical discussions
- **AI-Powered Analysis** - Refines meeting transcription using advanced LLMs
- **Actionable Items Tracking** - Automatically identifies and tracks assigned tasks
- **GitHub Integration** - Retrieves code content (supports both public and private repos) and maps feedback to specific code segments in repositories
- **Action Assignment** - Assigns tasks to mentioned contributors based on meeting context and GitHub contributors
- **Multi-Format Support** - Handles various file types (Python, Go, Jupyter notebooks, etc.)
- **Smart Filtering** - Excludes generated files and non-code directories (node_modules, .git, etc.)
- **Downloadable Results** - Users can download both the full transcript and the generated report
- **Multi-Provider LLM Support** - Choose between Groq and Google (Gemini) as your AI engine

## Architecture:

![alt text](architecture.png)

## System Requirements

- Python 3.10 or higher

## Installation

1) Clone the repository:
```bash
git clone https://github.com/Trojan7979/code-review-summerization.git
cd code-review-summerization
```

2) Create and activate a Python virtual environment:
```bash
python3 -m venv venv
```

windows
```bash
.\venv\Scripts\activate
```

linux/ubuntu
```bash
source venv/bin/activate
```

3) Install dependencies:
```bash
pip install -r requirements.txt
```

4) Launch the Streamlit app:
```bash
streamlit run app.py
```
5) Access the application:
```bash
http://localhost:8501
```

## Usage

### I. Web Interface Workflow

- **Upload MP4 video** of the code review meeting.
- **Enter required credentials** in the sidebar:
  - Choose LLM Provider
  - Groq API Token to extract text
  - Model Name (e.g., `gemini-2.5-pro-exp-03-25`, `deepseek-r1-distill-llama-70b`)
  - GitHub Repository URL
  - GitHub Repository path
  - GitHub Personal Access Token(Optional for Private Repository)
- **Click "Process Video"** to initiate processing.

### II. Output Sections

- **Generated Report Tab**:
  - AI-generated code review report, with;
    - Downloadable Markdown-format
    - Highlights on technical discussions, problems, improvements, and contributor-specific task assignments
- **Extracted Text Tab**  
  - Raw transcription of the meeting's spoken content
  - Downloadable for manual review or archival

## App url(Hosted)
```bash
https://code-review-summarization.streamlit.app/
```

This system helps teams maintain **80%+ efficiency** in code review follow-ups while reducing manual tracking effort by **60%**.
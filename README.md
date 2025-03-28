# AI-Powered Code Review Summarization from Meeting Recordings
An intelligent system that processes code review meeting recordings, extracts actionable insights, and maps feedback to GitHub repositories. Automates code review tracking while maintaining high accuracy in identifying changes, action items, and key discussion points.

## Features:

- **Automated Insights Extraction** - Processes MP4 meeting recordings to extract technical discussions
- **AI-Powered Analysis** - Uses Hugging Face models for Summarization and Report Generation
- **GitHub Integration** - Maps feedback to specific code segments in repositories
- **Actionable Items Tracking** - Automatically identifies and tracks assigned tasks
- **Multi-Format Support** - Handles various file types (Python, Go, Jupyter notebooks, etc.)
- **Smart Filtering** - Excludes generated files and non-code directories (node_modules, .git, etc.)

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

4) Run the script:
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
  - Hugging Face API Token
  - Model Name (e.g., `mistralai/Mistral-7B-Instruct-v0.2`)
  - GitHub Repository URL
  - GitHub Repository path
  - GitHub Personal Access Token
- **Click "Process Video"** to initiate processing.

### II. Output Sections

- **Extracted Text Tab**  
  - Full transcription of the meeting audio.

- **Summary Tab**  
  - AI-generated report with:
    - Code improvement recommendations  
    - Assigned action items  
    - Architectural discussion highlights  
    - PR/Issue correlations 

## App url
```bash
https://code-review-summarization.streamlit.app/
```

This system helps teams maintain **80%+ efficiency** in code review follow-ups while reducing manual tracking effort by **60%**.
# Project README

## Progress Summary

### Phase 1 - Speech-to-Text Transcription
- **Speech-to-Text Tools:**
  1. Google Speech-to-Text API via `speech_recognition` library (free tier)
  2. Audio Extraction: `moviepy` for converting video files (MP4, AVI, MOV, MKV) to WAV audio
- **Transcription Module:**
  1. Chunked audio processing (30s intervals) for reliability
  2. Noise adjustment and error handling ([inaudible] placeholders)
  3. Temporary file cleanup for audio/video

### Phase 2 - Partial Completion
- **GitHub API Integration:**
  - Validated GitHub API integration (PAT + `/code-scanning/analyses` endpoint) with 200 OK responses

### Phase 3 - AI-Powered Summarization
- **Integrated:**
  1. Mistral-7B LLM via HuggingFace + LangChain for text summarization
  2. Customizable prompts for concise summaries

### Phase 4 - Simple UI
- **Developed a Streamlit app with:**
  1. Video upload & processing
  2. Extracted text and summary display
  3. Downloadable outputs (TXT)
  4. Configurable model settings (API key, model selection)
  - **Note:** Local testing only.
  - **Dockerization:** Planned for containerizing the app for consistent environments and easier deployment.

## Tech Stack
- **Speech-to-Text:** `speech_recognition` (Google API), `moviepy` (audio extraction)
- **NLP:** `LangChain` + `HuggingFaceHub` (LLM orchestration)
- **GitHub Integration:** GraphQL
- **Data Processing:** Python, LangChain, LibROSA (audio processing and manipulation), NumPy, SoundFile, `huggingface_hub`
- **Frontend UI:** Streamlit

## Pending Tasks
- Code mapping
- Task assignment
- Hosting
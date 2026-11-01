# NexaBot AI User Guide

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Start Ollama and download the default model in PowerShell:
   `ollama pull qwen2.5:1.5b`
3. Activate the existing project environment and install Python dependencies:
   `python -m pip install -r requirements.txt`
4. From the project root, run `streamlit run app.py`.
5. Select an installed model and answer length in the sidebar. Ask a question
   or send a follow-up; NexaBot supplies recent chat history to the local model.
6. Use **New conversation**, **Clear chat**, or **Delete chat** to manage local
   history.

The model download needs internet once. The model runs locally after download.
If Ollama is unavailable or the selected model is missing, NexaBot displays
instructions rather than returning a canned answer.

# NexaBot AI

NexaBot AI is a local generative chatbot. It uses an open-source language model
through Ollama on your own Windows computer. Your prompts are sent to the local
Ollama service, not to a paid or cloud AI API.

The older TF-IDF and Logistic Regression intent model remains in the project for
experiments and analytics. It does not choose or limit chat responses.

## What is a Large Language Model?

A Large Language Model (LLM) is a neural network trained on large amounts of
text. It generates an answer by predicting text that fits the prompt and
conversation. It can explain topics, help with code, write drafts, and answer
many questions that are not listed in this project's intent data.

An LLM can still make mistakes or invent details. Check important facts and
never treat it as a substitute for professional advice.

## What is Ollama?

Ollama is a local program that downloads and runs open-source language models.
NexaBot communicates with Ollama's local HTTP API at
`http://127.0.0.1:11434` by default. Change the endpoint with the
`NEXABOT_OLLAMA_URL` environment variable.

The first Ollama and model downloads need an internet connection. Once the
software and model are downloaded, inference can run offline on your computer.

## How NexaBot generates answers

1. The Streamlit page reads the current chat history from SQLite.
2. NexaBot sends the latest messages and its assistant instructions to the
   selected local Ollama model.
3. Ollama generates text in a stream; NexaBot displays it as it arrives.
4. The completed user and assistant messages, including the model name, are
   saved in the existing SQLite database.

Recent turns are sent with follow-up questions so the model can resolve
references such as “Who created it?” The context is limited to the newest
messages to keep prompts manageable.

## Selected model and hardware

The default model is **`qwen2.5:1.5b`**. It is a compact instruction model
selected for CPU use and limited system memory. A smaller model uses less
storage and memory but may give less detailed or less reliable answers than a
larger model.

This computer reports about 13.8 GB of total memory. Before installing Ollama,
it had about 28.1 GB of free disk space; the downloaded model weights are about
986 MB. During local generation, available memory dropped below 1 GB, so close
other applications to reduce paging and improve speed. The model loaded and
generated answers on this computer's CPU, an AMD Ryzen 7 5825U. Actual speed
depends on CPU, available RAM, model quantization, and prompt size. A GPU is
not required.

## Install and run on Windows

Open PowerShell and run these steps from the existing project folder.

### 1. Install Ollama

Download and install the Windows version from
[ollama.com/download](https://ollama.com/download). Ollama may start
automatically after installation. If it does not, launch Ollama from the
Windows Start menu.

### 2. Download the model

In PowerShell, download the default model:

```powershell
ollama pull qwen2.5:1.5b
```

This model download is separate from the Python project. It needs internet
access once and uses local disk space.

To try a different model, download it with `ollama pull MODEL_NAME`, then
select it in NexaBot's sidebar. You can list downloaded models using:

```powershell
ollama list
```

### 3. Activate the existing Python environment

```powershell
cd "C:\Users\Alan Binoy\Desktop\NexaBotAI"
.\.venv\Scripts\Activate.ps1
```

If this project does not have a `.venv` folder yet, create it once before
activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Start NexaBot

Make sure Ollama is running and the model has been downloaded. Then run:

```powershell
streamlit run app.py
```

### 6. Run automated tests

```powershell
python -m pytest
```

The automated Ollama tests use mocks. They do not download a model or claim
that a mocked answer came from a real LLM.

## Change the model or endpoint

- Download another Ollama model using `ollama pull MODEL_NAME`; it will appear
  in the model selector in the sidebar.
- To change the default model, set `NEXABOT_OLLAMA_MODEL` before starting
  Streamlit. For example:

```powershell
$env:NEXABOT_OLLAMA_MODEL = "qwen2.5:1.5b"
streamlit run app.py
```

- To use a non-default local Ollama endpoint, set `NEXABOT_OLLAMA_URL`, for
  example `$env:NEXABOT_OLLAMA_URL = "http://127.0.0.1:11434"`.
- Temperature and answer length can be changed in the app sidebar.

## Connection errors

- **Ollama not installed:** install it from the official download page above.
- **Service is not running:** start Ollama from the Windows Start menu.
- **Model is missing:** run the `ollama pull ...` command shown in the sidebar.
- **Not enough memory:** close other applications or download a smaller model.
- **Slow or timed out:** shorten the question or use a smaller model.

NexaBot displays these errors instead of substituting fixed responses.

## Existing files and data

- `app.py`: Streamlit chat interface.
- `src/ollama_client.py`: local Ollama HTTP API, streaming, history limits,
  and clear error messages.
- `src/database.py`: SQLite storage and safe migration of existing message
  tables to include model names.
- `data/intents.json`: examples for the optional intent-classification
  experiment, not the generative assistant's allowed question list.
- `data/knowledge_base.json`: retained local reference data.
- `tests/`: automated tests for preprocessing, the optional classifier,
  database migration, and mocked Ollama interactions.

## Privacy and limitations

Inference is local when using the default Ollama endpoint. Chat history remains
in `data/chat_history.db`. Model installation and updates require internet
access, but normal inference does not. Do not configure this app to use an
endpoint you do not trust.

NexaBot is not ChatGPT and does not have unlimited knowledge. It has no live
internet access, can make mistakes, and may run slowly on CPU. Verify
important information and generated code before relying on it.

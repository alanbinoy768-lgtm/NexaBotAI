# NexaBot AI Architecture

The local chat path is:

1. Streamlit reads the selected conversation and its recent messages from SQLite.
2. The Ollama client prepends NexaBot's system instructions and applies a
   bounded recent-message history.
3. The selected model generates an answer using Ollama's local `/api/chat`
   endpoint.
4. Streamed text is shown as it arrives.
5. User and assistant messages, including the model name, are saved in SQLite.

The legacy NLTK, TF-IDF, and Logistic Regression pipeline remains available
for classifier experiments. It is not used to select or limit generated chat
answers. Local knowledge retrieval remains as reference data and is not used
to fake LLM responses.

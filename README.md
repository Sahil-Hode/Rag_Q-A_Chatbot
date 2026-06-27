# 🤖 Simple PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly based on the content of your own PDF documents**. Instead of relying on an LLM's general knowledge (and risking hallucinated answers), this app retrieves the most relevant passages from your PDFs first, then asks the LLM to answer using only that retrieved context — with full transparency into which source chunks were used.

---

## 🧠 Why RAG?

A standard LLM chatbot answers from its training data, which means it can:
- Hallucinate facts that sound plausible but are wrong
- Have no knowledge of your private/internal documents
- Give answers it can't trace back to a source

This app solves all three by combining **retrieval** (finding the right text from your PDFs) with **generation** (having the LLM phrase an answer from that text) — and showing you the exact source chunks behind every answer.

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| LLM (generation) | Mistral AI — `mistral-large-latest` |
| Embeddings | Mistral AI Embeddings |
| Vector Database | ChromaDB (local, persistent on disk) |
| Orchestration | LangChain (`langchain`, `langchain-mistralai`, `langchain-community`) |
| Frontend UI | Streamlit |
| PDF Parsing | `PyPDFLoader` (LangChain Community) |

---

## 🏗 Architecture Overview

The system is split into two independent phases that never run at the same time:

```
┌─────────────────────┐         ┌──────────────────────────┐
│   INGESTION PHASE    │         │   QUERY PHASE (Chat UI)   │
│   (run once/offline) │         │   (run repeatedly)        │
│                       │         │                           │
│   ingest.py           │         │   app.py + utils.py       │
└─────────────────────┘         └──────────────────────────┘
         │                                    │
         ▼                                    ▼
   PDFs → Chunks → Embeddings        Question → Embedding → Search
         │                                    │
         ▼                                    ▼
   Stored in chroma_db/  ───────────►  Retrieved from chroma_db/
                                               │
                                               ▼
                                    Context + Question → Mistral LLM
                                               │
                                               ▼
                                          Answer + Sources
```

**Key design decision:** ingestion and querying are decoupled. You only need to re-run `ingest.py` when your PDFs change — not every time you ask a question. This keeps the chat experience fast since embeddings are computed once and reused.

---

## 📂 Project Structure

```text
rag-chatbot/
│
├── data/                   # Place your PDF files here
│   └── python.pdf          # Example PDF
│
├── chroma_db/              # Auto-generated vector database (created by ingest.py)
│
├── .env                    # Your actual API key (not committed to git)
├── .env.example            # Template showing required env vars
├── requirements.txt        # Python dependencies
├── ingest.py               # Loads PDFs → chunks → embeds → stores in ChromaDB
├── utils.py                # Core RAG logic: retrieval + prompt + generation
├── app.py                  # Streamlit chat UI
└── README.md               # This file
```

---

## 🔍 How It Actually Works (Code-Level Walkthrough)

### Phase 1 — Ingestion (`ingest.py`)

This script is run **manually, once**, whenever you add or change PDFs.

1. **`load_documents()`** scans the `data/` folder for every `*.pdf` file using `glob`, and loads each one with `PyPDFLoader`. Each page of each PDF becomes a separate LangChain `Document` object. If no PDFs are found, it prints a message and exits early rather than failing silently.

2. **Chunking**: All loaded pages are split using `RecursiveCharacterTextSplitter` with:
   - `chunk_size = 500` characters
   - `chunk_overlap = 100` characters

   The 100-character overlap is intentional — it prevents a sentence or idea from being cut cleanly in half between two chunks, which would otherwise hurt retrieval quality.

3. **Embedding + Storage**: `MistralAIEmbeddings()` converts each chunk into a vector. `Chroma.from_documents(...)` then embeds and persists everything directly to `chroma_db/` in one call — there's no separate "save" step, persistence happens automatically via the `persist_directory` argument.

**Run it with:**
```bash
python ingest.py
```

You'll see console output showing each PDF being loaded, the total page count, the number of chunks created, and a final `Ingestion complete! Data is ready for retrieval.` message.

---

### Phase 2 — Retrieval & Generation (`utils.py` + `app.py`)

This is the live chat logic, triggered every time a user submits a question.

**`get_vector_store()`** (in `utils.py`):
- Checks whether `chroma_db/` exists on disk. If not, it raises a `FileNotFoundError` with a clear message telling you to run `ingest.py` first — this is caught and displayed nicely in the UI rather than crashing the app.
- If it exists, it loads the persisted Chroma database using the same embedding function used at ingestion time.

**`answer_question(question: str)`** (in `utils.py`) — the heart of the RAG pipeline:

1. **Retrieve**: Calls `vector_store.as_retriever(search_kwargs={"k": 3})` to fetch the **top 3 most semantically similar chunks** to the user's question.
2. **Build context**: Joins those 3 chunks into a single string, separated by double newlines.
3. **Strict prompt**: Constructs a prompt via `ChatPromptTemplate` that explicitly instructs the model: *answer ONLY from the given context*, and if the answer isn't present, reply with an exact fallback sentence — *"I couldn't find that information in the provided document."* This hard-coded fallback is what prevents hallucination; the model isn't allowed to guess.
4. **Generate**: Pipes the prompt into `ChatMistralAI(model="mistral-large-latest", temperature=0)`. Temperature is set to `0` deliberately — this minimizes creative/random variation since the goal is factual, document-grounded answers, not creative writing.
5. **Return**: Returns a tuple of `(answer_text, source_documents)` — both the generated answer and the raw chunks that were retrieved, so the UI can show its "receipts."

**`app.py`** (Streamlit UI):
- Maintains conversation history in `st.session_state.messages` so the chat persists across reruns within a session (Streamlit reruns the whole script on every interaction).
- On each new question, calls `answer_question()` inside a `st.spinner` for a loading indicator, then renders the answer.
- Every assistant response that has source chunks attached gets a **"View Retrieved Source Chunks"** expander, listing each chunk's source filename (extracted from `doc.metadata['source']`) and its raw text — this is what gives the app its transparency/explainability.
- Wraps the call in `try/except`, specifically catching `FileNotFoundError` (the "you forgot to run ingest.py" case) separately from generic exceptions, so error messages are actually useful instead of a raw stack trace.

---

## 🚀 Setup & Installation

### 1. Clone and enter the project directory
```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Create and activate a virtual environment
Isolating dependencies avoids version conflicts with other Python projects on your machine.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it — Linux/macOS
source venv/bin/activate

# Activate it — Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your Mistral API key
```bash
cp .env.example .env
```
Open `.env` and set:
```env
MISTRAL_API_KEY=your_actual_key_here
```
> **Note:** Both `ingest.py` and `utils.py` call `load_dotenv(override=True)`. The `override=True` flag means values in your `.env` file will take priority over any conflicting environment variables already set in your system/shell — useful if you have a stale `MISTRAL_API_KEY` set globally and want `.env` to always win.

Get a Mistral API key at [console.mistral.ai](https://console.mistral.ai) if you don't have one yet.

### 5. Add your PDFs
Place one or more `.pdf` files inside the `data/` folder.

---

## 💡 Usage

### Step 1 — Run ingestion (only needed once per set of PDFs)
```bash
python ingest.py
```
Expected console output:
```
Starting ingestion process...
Loading data/python.pdf...
Loaded 42 total pages.
Splitting documents into chunks (size=500, overlap=100)...
Created 187 document chunks.
Generating embeddings and storing in ChromaDB at 'chroma_db'...
Ingestion complete! Data is ready for retrieval.
```

### Step 2 — Launch the chatbot
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

### Step 3 — Ask questions
- Type a question in the chat input box.
- The app retrieves the 3 most relevant chunks, sends them + your question to Mistral, and displays the answer.
- Click **"View Retrieved Source Chunks"** under any answer to see exactly which text the model used — useful for verifying accuracy or catching when the document genuinely doesn't contain the answer.

### Step 4 — Add more PDFs later
Drop new PDFs into `data/` and re-run `python ingest.py`. The new content will be embedded and added to the existing `chroma_db/` store.

---

## ⚙️ Configuration Reference

| Setting | Location | Default | Notes |
|---|---|---|---|
| Chunk size | `ingest.py` | 500 chars | Increase for more context per chunk, decrease for finer-grained retrieval |
| Chunk overlap | `ingest.py` | 100 chars | Prevents context loss at chunk boundaries |
| Top-K retrieved chunks | `utils.py` (`search_kwargs={"k": 3}`) | 3 | Increase if answers seem to be missing context from your PDFs |
| LLM model | `utils.py` | `mistral-large-latest` | Swap for a smaller/cheaper Mistral model if cost is a concern |
| Temperature | `utils.py` | 0 | Keep at 0 for factual consistency; raising it adds variability |

---

## 🐛 Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Illegal header value b'Bearer '` | `MISTRAL_API_KEY` is empty or missing | Confirm `.env` exists (not just `.env.example`), is named exactly `.env`, and the key has no quotes around it |
| `No PDF files found in 'data' directory` | No `.pdf` files in `data/`, or wrong file extension | Confirm files are directly inside `data/` and end in `.pdf` (not `.PDF` uppercase, depending on OS) |
| `Database directory 'chroma_db' not found` | `ingest.py` hasn't been run yet | Run `python ingest.py` once before starting `streamlit run app.py` |
| Answers seem irrelevant or generic | Retrieval is pulling the wrong chunks | Try increasing `k` in `utils.py`, or reduce `CHUNK_SIZE` in `ingest.py` for more precise chunks |
| App says it "couldn't find that information" for something that IS in the PDF | The chunk containing the answer wasn't in the top-3 retrieved | Increase `k`, or check that the PDF text extracted cleanly (scanned/image-based PDFs won't extract text at all) |

---

## 📌 Limitations

- **Text-only PDFs**: `PyPDFLoader` extracts text directly from the PDF — scanned/image-based PDFs with no embedded text layer will yield empty or broken extraction (no OCR is performed).
- **No persistent chat memory across sessions**: Conversation history lives in Streamlit's session state and resets when the server restarts or the browser session ends.
- **Single embedding/LLM provider**: Tightly coupled to Mistral AI; swapping providers means updating both `ingest.py` and `utils.py`.

---

## 🗺 Possible Extensions

- Add OCR (e.g., via `pytesseract`) to support scanned PDFs
- Support additional file types (`.docx`, `.txt`, `.md`) via other LangChain loaders
- Add a "clear chat history" button in the Streamlit UI
- Persist chat history to disk/database for cross-session memory
- Add citation highlighting (show exactly which sentence within a chunk was used)

---

## 📄 License

Add your preferred license here (MIT, Apache 2.0, etc.)# Rag_Q-A_Chatbot

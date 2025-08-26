# Agentic RAG Chatbot

A simple, fast **Retrieval‑Augmented Generation (RAG)** chatbot that answers questions from your **PDF documents** using **LangChain**, **Groq**, and **Pinecone**, with a **FastAPI** backend and a clean chat UI.

---

## ✨ Features

* 🤖 **RAG answers** powered by LangChain + Groq
* 📄 **PDF processing** with automatic text chunking
* 🔍 **Vector search** using Pinecone
* ⚡ **FastAPI backend** with real‑time chat
* 💬 **Modern, minimal UI** (HTML/CSS/JS)
* 🔒 **Secure secrets** via `.env`

---

## 🚀 Quick Start

### 1) Clone

```bash
git clone <your-repo-url>
cd newrag
```

### 2) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install \
  fastapi uvicorn python-multipart jinja2 \
  langchain-groq langchain langgraph \
  pinecone-client langchain-community \
  langchain-text-splitters langchain-core \
  huggingface-hub sentence-transformers \
  PyPDF2 python-dotenv
```

> Tip: If you prefer a `requirements.txt`, you can generate one later with `pip freeze > requirements.txt`.

### 4) Set environment variables

Copy the example file and fill in your keys.

**Windows:**

```bash
copy .env.example .env
```

**Linux/Mac:**

```bash
cp .env.example .env
```

Edit `.env` and set your values:

```env
GROQ_API_KEY=your_actual_groq_api_key
PINECONE_API_KEY=your_actual_pinecone_api_key
PINECONE_API_ENV=us-east-1
HUGGINGFACEHUB_API_TOKEN=your_actual_huggingface_token
HF_TOKEN=your_actual_huggingface_token
```

### 5) Point to your PDF

Open `rag_agent.py` and set the PDF path:

```python
pdf_file_path = "path/to/your/file.pdf"
```

### 6) Run the app

```bash
python main.py
```

Now open: **[http://127.0.0.1:8007](http://127.0.0.1:8007)**

---

## 🧭 Project Structure

```
newrag/
├─ main.py               # FastAPI app entrypoint
├─ rag_agent.py          # RAG agent implementation
├─ templates/
│  └─ chat.html          # Chat UI (Jinja2 template)
├─ static/               # CSS/JS/assets
├─ .env                  # Your secrets (DO NOT COMMIT)
├─ .env.example          # Example env file
├─ .gitignore            # Git ignore rules
└─ README.md             # This file
```

---

## 🧪 API Endpoints

* `GET /` – Chat UI
* `POST /api/chat` – Chat endpoint
* `GET /api/health` – Health check

### Example: send a chat message

```bash
curl -X POST http://127.0.0.1:8007/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What does the PDF say about X?"}'
```

---

## 🔐 Security Notes

*  Never commit `.env` (keep your API keys safe)
*  Use `.env.example` for documentation only
*  API keys are loaded from environment variables
*  `.gitignore` prevents committing secrets

Suggested `.gitignore` entries (if needed):

```
.env
venv/
__pycache__/
*.pyc
```

---

## 🧰 Technologies

* **Backend:** FastAPI, Uvicorn
* **AI / LLM:** LangChain, Groq, Hugging Face
* **Vector DB:** Pinecone
* **Frontend:** HTML, CSS, JavaScript (Jinja2 template)
* **Docs & Parsing:** PyPDF2, LangChain loaders/splitters

---

## 🩺 Troubleshooting

* **`ModuleNotFoundError: No module named 'uvicorn'`** → Activate your venv and run `pip install uvicorn`.
* **Server starts but page won’t load** → Make sure the app is running on `http://127.0.0.1:8007` and no other app is using that port.
* **Pinecone errors** → Check `PINECONE_API_KEY` and `PINECONE_API_ENV` values.
* **No answers from PDF** → Confirm `pdf_file_path` is correct and readable; restart the server after changes.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes & add tests if possible
4. Open a pull request


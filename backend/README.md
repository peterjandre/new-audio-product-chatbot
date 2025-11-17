# Backend - Audio Products Chatbot API

FastAPI backend for the RAG-powered audio products chatbot.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export FAISS_INDEX_PATH="data/faiss_index.index"
export CORPUS_PATH="data/gearspace_corpus.json"
export OPENAI_MODEL="gpt-3.5-turbo"
```

3. Run the server:
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload --port 8000
```

## Vercel Deployment

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
cd backend
vercel
```

3. Set environment variables in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `FAISS_INDEX_PATH` (default: `data/faiss_index.index`)
   - `CORPUS_PATH` (default: `data/gearspace_corpus.json`)
   - `OPENAI_MODEL` (default: `gpt-3.5-turbo`)

4. Upload data files:
   - Upload `data/faiss_index.index`
   - Upload `data/faiss_index_metadata.json` (if exists)
   - Upload `data/gearspace_corpus.json`

Note: Vercel has a 50MB limit for serverless functions. If your data files exceed this, consider using external storage (S3, etc.) and downloading them at runtime.

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `POST /query` - Query the RAG engine

## Project Structure

```
backend/
├── api/
│   └── index.py          # Vercel serverless function wrapper
├── scripts/               # RAG engine scripts
├── data/                  # Data files (FAISS index, corpus)
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
└── vercel.json           # Vercel configuration
```


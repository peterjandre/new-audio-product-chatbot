# Audio Products Chatbot - RAG-Powered Search Engine

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about audio production gear by searching through posts from Gearspace.com.

## Project Structure

This project is split into two separate applications for deployment:

```
.
├── backend/          # FastAPI backend (Python)
│   ├── api/         # Vercel serverless functions
│   ├── scripts/     # RAG engine scripts
│   ├── data/        # FAISS index and corpus data
│   ├── app.py       # FastAPI application
│   └── vercel.json  # Vercel configuration
│
├── frontend/        # TypeScript frontend
│   ├── src/         # TypeScript source files
│   ├── public/      # Static files (HTML, CSS, compiled JS)
│   ├── package.json # Node.js dependencies
│   └── vercel.json  # Vercel configuration
│
└── data/            # Original data files (for reference)
```

## Quick Start

### Backend Setup

See [backend/README.md](backend/README.md) for detailed instructions.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup

See [frontend/README.md](frontend/README.md) for detailed instructions.

```bash
cd frontend
npm install
npm run build
```

## Vercel Deployment

### Deploy Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Deploy to Vercel:
```bash
vercel
```

3. Set environment variables in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `FAISS_INDEX_PATH` (default: `data/faiss_index.index`)
   - `CORPUS_PATH` (default: `data/gearspace_corpus.json`)
   - `OPENAI_MODEL` (default: `gpt-3.5-turbo`)

4. Upload data files to Vercel (or use external storage for large files)

5. Note the deployment URL (e.g., `https://your-backend.vercel.app`)

### Deploy Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Set environment variable in Vercel dashboard:
   - `API_BASE_URL` - Set to your backend URL (e.g., `https://your-backend.vercel.app`)

3. Deploy to Vercel:
```bash
vercel
```

## How It Works

1. **Data Collection:** The system scrapes RSS feeds from Gearspace.com and generates concise summaries of each post.

2. **Embedding Generation:** OpenAI's embedding model converts the text summaries into high-dimensional vectors for semantic search.

3. **Vector Index:** FAISS (Facebook AI Similarity Search) creates an efficient index for fast similarity search across thousands of posts.

4. **Query Processing:** When you ask a question, the system:
   - Converts your question into an embedding vector
   - Searches the FAISS index for the most relevant posts
   - Retrieves the top-k most similar posts
   - Uses OpenAI GPT to generate a contextual answer based on the retrieved posts

5. **Response Generation:** The LLM synthesizes information from multiple sources to provide a comprehensive answer with citations.

## Technologies Used

- **Backend:** FastAPI (Python), Mangum (for Vercel serverless)
- **LLM Provider:** OpenAI GPT-3.5/GPT-4
- **Embeddings:** OpenAI text-embedding-ada-002
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Frontend:** TypeScript, HTML5, CSS3
- **Deployment:** Vercel (serverless functions + static hosting)

## Development

### Building the RAG Index

1. Generate embeddings:
```bash
python scripts/generate_embeddings.py --corpus data/gearspace_corpus.json --cache data/embeddings_cache.json
```

2. Build FAISS index:
```bash
python scripts/build_faiss_index.py --cache data/embeddings_cache.json --index data/faiss_index.index
```

3. Copy data files to backend:
```bash
cp -r data/* backend/data/
```

## License

This project is for portfolio purposes.

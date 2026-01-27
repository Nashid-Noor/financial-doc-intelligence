# Document Intelligence Platform


A RAG (Retrieval-Augmented Generation) system for intelligent document analysis. Designed to handle complex documents, tables, and reasoning.

## Key Features

- **RAG Architecture:** Retrieves relevant chunks from financial documents to answer user queries.
- **Hybrid Search:** Combines semantic search (MiniLM) with keyword matching (BM25) for precision.
- **Citations:** Answers include specific page numbers and source text snippets.

## Quick Start


### Installation

```bash
# Clone the repository
git clone https://github.com/Nashid-Noor/financial-doc-intelligence.git
cd financial-doc-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup Environment
cp .env.example .env
# Edit .env and add your HF_API_KEY
```

### Running the Application

#### Option 1: Quick Start (Recommended)
The easiest way to run everything (API + UI) is using the helper script:

```bash
# Make executable first
chmod +x run.sh

# Start everything
./run.sh all
```

#### Option 2: Manual Start

```bash
cd src/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

#### 3. Access the Application

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API ReDoc**: http://localhost:8000/redoc

## Configuration

### RAG Configuration (`configs/rag_config.yaml`)

```yaml
retrieval:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  chunk_overlap: 50
```





## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```


## Docker Deployment

```bash
# Build image
docker build -t financial-doc-intelligence .

# Run container
docker run -p 8000:8000 -p 8501:8501 financial-doc-intelligence
```

### Render Deployment

The project includes a `render.yaml` file for easy deployment on [Render.com](https://render.com).
1. Create a Key-Value Secret file or simple environment variable for `HF_API_KEY`.
2. Connect your repository to Render.
3. Select "Web Service" and use the Docker runtime.








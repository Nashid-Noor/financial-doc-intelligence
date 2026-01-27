# Financial Document Intelligence Platform


A RAG (Retrieval-Augmented Generation) system specialized for SEC filings and complex financial documents. Designed to handle tabular data, numerical reasoning, and cross-document analysis.

## Key Features

- **Fine-Tuning Support:** Capable of fine-tuning Llama 3.1 8B on financial datasets (tested with FinQA/TAT-QA formats).
- **Hybrid Search:** Combines dense vectors (Qdrant) with sparse keyword matching (BM25) to catch both semantic meaning and specific terms.
- **Table-Aware Processing:** Preserves table structure during chunking to allow for accurate data extraction.
- **Citations:** Every answer includes page numbers and source links.

## Quick Start


### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-doc-intelligence.git
cd financial-doc-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
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

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4
```

## Training Data Compatibility

The platform is designed to work with standard financial datasets:

| Dataset | Description |
|---------|-------------|
| FinQA | Numerical reasoning over financial reports |
| TAT-QA | Hybrid tabular and textual Q&A |
| Custom | Your own SEC filing Q&A pairs |

## Evaluation

Run the evaluation suite:

```bash
python -m pytest tests/ -v

# Run specific evaluation
python src/evaluation/run_eval.py --dataset finqa --split test
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








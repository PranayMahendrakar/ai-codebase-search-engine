# AI Codebase Search Engine

> Semantic search for codebases using local open-source embedding models — no API keys, no internet required.

[![CI](https://github.com/PranayMahendrakar/ai-codebase-search-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/PranayMahendrakar/ai-codebase-search-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-brightgreen)

---

## What It Does

Point it at any codebase and ask questions in plain English — it finds the most semantically relevant functions, classes, and code blocks.

**Example:**

```bash
python search_engine.py --path ./my_project --query "authentication middleware"
```

**Output:**
```
======================================================================
  Search Results for: "authentication middleware"
======================================================================

[1] Score: 0.9124  [##################  ]
    File    : src/middleware/auth.py
    Function: verify_jwt_token
    Class   : AuthMiddleware
    Lines   : 45 - 72
    Type    : function

    class AuthMiddleware:
        """Middleware that verifies JWT tokens on every request."""
        def verify_jwt_token(self, token: str) -> dict:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            ...

[2] Score: 0.8756  [#################   ]
    File    : src/routes/login.py
    Function: authenticate_user
    Lines   : 12 - 35
    Type    : function

    def authenticate_user(username, password):
        """Authenticate user credentials against database."""
        ...
```

---

## Features

- **100% local** — uses HuggingFace Transformers offline, no API keys needed
- **Semantic understanding** — finds `authenticate_user` when you search "login logic"
- **Multi-language** — Python (AST), JS/TS, Go, Rust, Java, C++, and 30+ more
- **Smart parsing** — extracts functions, classes, methods with docstrings
- **Index caching** — rebuilds only when files change (fast repeated queries)
- **Small model** — `all-MiniLM-L6-v2` (~80MB, fits in GitHub Actions free tier)
- **CLI + Python API** — use from terminal or import in your own scripts
- **JSON output** — pipe results to other tools

---

## Architecture

```
your codebase
     |
     v
┌─────────────┐     ┌──────────────────────────────────┐
│ code_parser │ --> │ Chunks: functions, classes, files │
└─────────────┘     └──────────────────────────────────┘
                                   |
                                   v
                        ┌──────────────────┐
                        │   embedder.py    │ <── sentence-transformers
                        │  (all-MiniLM)    │     (local, no API key)
                        └──────────────────┘
                                   |
                              float32 vectors
                                   |
                                   v
                     ┌─────────────────────────┐
                     │   .code_index.pkl       │ <── cached on disk
                     │   .code_metadata.json   │
                     └─────────────────────────┘
                                   |
                        cosine similarity search
                                   |
                                   v
                          Ranked results with
                          file + function + score
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/PranayMahendrakar/ai-codebase-search-engine
cd ai-codebase-search-engine

# Install CPU-only PyTorch first (saves ~1.5GB vs default)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest
pip install -r requirements.txt
```

### 2. Search

```bash
# Basic search
python search_engine.py --path /path/to/your/project --query "database connection pool"

# Get more results
python search_engine.py --path . --query "error handling" --top-k 10

# Force re-index after big changes
python search_engine.py --path . --query "JWT validation" --rebuild

# Save results to JSON
python search_engine.py --path . --query "rate limiting" --output results.json
```

### 3. Use as Python Library

```python
from search_engine import CodebaseSearchEngine

# Initialize engine
engine = CodebaseSearchEngine("/path/to/your/project")

# Build the index (cached automatically)
engine.build_index()

# Search!
results = engine.search("authentication middleware", top_k=5)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"File : {result['file']}")
    print(f"Func : {result.get('function', 'N/A')}")
    print(f"Code :\n{result['text'][:300]}")
    print("---")
```

---

## CLI Reference

```
usage: search_engine.py [-h] [--path PATH] --query QUERY [--top-k TOP_K]
                        [--rebuild] [--model MODEL] [--output OUTPUT]

options:
  --path,  -p    Path to codebase directory (default: current directory)
  --query, -q    Search query - natural language or code snippet (required)
  --top-k, -k    Number of results to return (default: 5)
  --rebuild      Force rebuild index, ignore cache
  --model, -m    HuggingFace model name (default: all-MiniLM-L6-v2)
  --output,-o    Save results to JSON file
```

---

## Supported Models (all local, no API key)

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | ~80MB | Fast | **Default - general code** |
| `sentence-transformers/all-MiniLM-L2-v2` | ~25MB | Fastest | Memory-constrained |
| `microsoft/codebert-base` | ~500MB | Medium | Code-specialized search |
| `sentence-transformers/all-mpnet-base-v2` | ~420MB | Slower | Higher accuracy |

All models download once and cache in `~/.cache/huggingface/`.

---

## Supported Languages

Python, JavaScript, TypeScript, Java, C++, C, Go, Rust, Ruby, PHP, C#, Swift, Kotlin, Scala, Shell/Bash, SQL, HTML, CSS, YAML, JSON, Markdown, R, Lua, Dart, Elixir, and more.

---

## How It Works

**Step 1 — Parse**
The `CodeParser` scans your codebase:
- Python files: uses the `ast` module to extract functions/classes with their docstrings, signatures, and line ranges
- JS/TS: regex extraction of functions, arrow functions, classes
- Go/Rust: regex extraction of `fn`/`func` declarations
- All others: sliding window chunking (40 lines, 10-line overlap)

**Step 2 — Embed**
Each code chunk is converted to a 384-dimensional float vector using `all-MiniLM-L6-v2`, a small but powerful sentence embedding model that runs locally via HuggingFace Transformers.

**Step 3 — Index**
All embeddings are stored as a NumPy float32 matrix on disk (`.code_index.pkl`). A hash of file modification times determines cache validity.

**Step 4 — Search**
Your query is embedded with the same model. Cosine similarity is computed against all indexed chunks. Top-K results are returned with scores, file paths, function names, and line numbers.

---

## GitHub Actions CI

The CI pipeline (`.github/workflows/ci.yml`) runs:
- **Unit tests** across Python 3.9, 3.10, 3.11
- **Linting** with flake8, black, isort
- **Demo search** on the project itself (main branch only)

Uses `actions/cache` for HuggingFace model caching, and CPU-only PyTorch to stay within GitHub Actions free tier limits.

---

## Project Structure

```
ai-codebase-search-engine/
├── search_engine.py         # Main CLI + CodebaseSearchEngine class
├── embedder.py              # CodeEmbedder - local HuggingFace inference
├── code_parser.py           # CodeParser - multi-language code chunking
├── requirements.txt         # Dependencies
├── tests/
│   └── test_search_engine.py  # Full test suite (mock embedder, no model download)
└── .github/
    └── workflows/
        └── ci.yml           # GitHub Actions CI/CD pipeline
```

---

## Running Tests

```bash
# Install test deps
pip install pytest pytest-cov

# Run tests (uses mock embedder - no model download needed)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with HuggingFace Transformers | No API keys | Runs 100% offline*

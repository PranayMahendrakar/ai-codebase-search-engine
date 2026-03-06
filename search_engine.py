"""
AI Codebase Search Engine
Semantic search over codebases using local open-source embedding models.
No API keys required - runs entirely offline using HuggingFace Transformers.
"""

import json
import pickle
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from embedder import CodeEmbedder
from code_parser import CodeParser

INDEX_FILE = ".code_index.pkl"
METADATA_FILE = ".code_metadata.json"

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".sql", ".html", ".css", ".yaml", ".yml", ".json",
    ".md", ".txt", ".r", ".m", ".lua", ".dart", ".ex", ".exs"
}


class CodebaseSearchEngine:
    """
    Semantic search engine for codebases.
    Indexes code files using local embedding models and performs
    similarity search using cosine similarity.
    """

    def __init__(self, codebase_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.codebase_path = Path(codebase_path).resolve()
        self.embedder = CodeEmbedder(model_name)
        self.parser = CodeParser()
        self.index: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.index_path = self.codebase_path / INDEX_FILE
        self.metadata_path = self.codebase_path / METADATA_FILE

    def _get_codebase_hash(self) -> str:
        """Generate a hash of file modification times for cache invalidation."""
        hasher = hashlib.md5()
        for file_path in sorted(self.codebase_path.rglob("*")):
            if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
                stat = file_path.stat()
                hasher.update(f"{file_path}:{stat.st_mtime}".encode())
        return hasher.hexdigest()

    def _load_index(self) -> bool:
        """Load existing index from disk if available and valid."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    cached = pickle.load(f)
                with open(self.metadata_path, "r") as f:
                    meta = json.load(f)
                current_hash = self._get_codebase_hash()
                if cached.get("hash") == current_hash:
                    self.index = cached["embeddings"]
                    self.metadata = meta["chunks"]
                    print(f"[Cache] Loaded existing index ({len(self.metadata)} chunks).")
                    return True
            except Exception as e:
                print(f"[Cache] Could not load index: {e}")
        return False

    def _save_index(self, codebase_hash: str):
        """Persist the index to disk."""
        with open(self.index_path, "wb") as f:
            pickle.dump({"hash": codebase_hash, "embeddings": self.index}, f)
        with open(self.metadata_path, "w") as f:
            json.dump({"chunks": self.metadata}, f, indent=2)
        print(f"[Cache] Index saved to {self.index_path}")

    def build_index(self, force_rebuild: bool = False):
        """
        Scan codebase, parse code chunks, embed them, and build the search index.
        Uses cached index if available and up-to-date.
        """
        if not force_rebuild and self._load_index():
            return

        print(f"[Index] Scanning codebase at: {self.codebase_path}")
        all_chunks = []
        file_count = 0

        for file_path in self.codebase_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix in SUPPORTED_EXTENSIONS
                and not any(p.startswith(".") for p in file_path.parts)
                and "node_modules" not in file_path.parts
                and "__pycache__" not in file_path.parts
                and "venv" not in file_path.parts
                and ".git" not in file_path.parts
            ):
                chunks = self.parser.parse_file(file_path, self.codebase_path)
                all_chunks.extend(chunks)
                file_count += 1

        if not all_chunks:
            print("[Index] No supported files found in the codebase.")
            return

        print(f"[Index] Found {file_count} files -> {len(all_chunks)} code chunks.")
        print("[Index] Embedding chunks (this may take a moment)...")

        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(texts)

        self.index = np.array(embeddings, dtype=np.float32)
        self.metadata = all_chunks

        codebase_hash = self._get_codebase_hash()
        self._save_index(codebase_hash)
        print(f"[Index] Index built with {len(all_chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search on the indexed codebase.

        Args:
            query: Natural language or code query string
            top_k: Number of top results to return

        Returns:
            List of result dicts with file, function, score, and snippet
        """
        if self.index is None or len(self.metadata) == 0:
            print("[Search] Index is empty. Run build_index() first.")
            return []

        query_embedding = self.embedder.embed(query)
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self.index, axis=1)
        query_norm = np.linalg.norm(query_vec)
        similarities = np.dot(self.index, query_vec) / (norms * query_norm + 1e-10)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.metadata[idx].copy()
            chunk["score"] = float(similarities[idx])
            results.append(chunk)

        return results


def display_results(results: List[Dict], query: str):
    """Pretty-print search results to the terminal."""
    print("\n" + "=" * 70)
    print(f'  Search Results for: "{query}"')
    print("=" * 70)

    if not results:
        print("  No results found.")
        return

    for i, result in enumerate(results, 1):
        score_pct = int(result["score"] * 20)
        score_bar = "#" * score_pct
        print(f"\n[{i}] Score: {result['score']:.4f}  [{score_bar:<20}]")
        print(f"    File    : {result['file']}")
        if result.get("function"):
            print(f"    Function: {result['function']}")
        if result.get("class"):
            print(f"    Class   : {result['class']}")
        print(f"    Lines   : {result.get('start_line', '?')} - {result.get('end_line', '?')}")
        print(f"    Type    : {result.get('chunk_type', 'code')}")
        print()
        snippet = result["text"]
        lines = snippet.strip().split("\n")
        preview = "\n    ".join(lines[:12])
        print(f"    {preview}")
        if len(lines) > 12:
            print(f"    ... ({len(lines) - 12} more lines)")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="AI Codebase Search Engine - Semantic search using local models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_engine.py --path ./my_project --query "authentication middleware"
  python search_engine.py --path . --query "database connection" --top-k 10
  python search_engine.py --path ./src --query "error handling" --rebuild
  python search_engine.py --path . --query "JWT token validation"
        """
    )
    parser.add_argument("--path", "-p", default=".", help="Path to codebase (default: current directory)")
    parser.add_argument("--query", "-q", required=True, help="Search query (natural language or code snippet)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index (ignore cache)")
    parser.add_argument(
        "--model", "-m",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name for embeddings"
    )
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    engine = CodebaseSearchEngine(args.path, model_name=args.model)
    engine.build_index(force_rebuild=args.rebuild)
    results = engine.search(args.query, top_k=args.top_k)
    display_results(results, args.query)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Output] Results saved to {args.output}")


if __name__ == "__main__":
    main()

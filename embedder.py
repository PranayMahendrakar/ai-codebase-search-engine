"""
Code Embedder Module
Generates semantic embeddings for code chunks using local HuggingFace models.
Supports multiple small models optimized for code understanding.
No API keys required - all models run locally.
"""

import os
from typing import List, Union
from pathlib import Path

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import numpy as np


# Small, fast models ideal for code semantic search (all run locally)
RECOMMENDED_MODELS = {
    # Best overall - fast, accurate, 80MB
    "default": "sentence-transformers/all-MiniLM-L6-v2",
    # Code-optimized embeddings
    "code": "microsoft/codebert-base",
    # Multilingual support
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # Ultra-fast, smallest model
    "tiny": "sentence-transformers/all-MiniLM-L2-v2",
    # Better accuracy, slightly larger
    "large": "sentence-transformers/all-mpnet-base-v2",
}

BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512


class CodeEmbedder:
    """
    Local embedding model for code semantic search.
    Uses HuggingFace Transformers to generate dense vector representations.
    Works entirely offline - no API keys needed.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model. Tries sentence-transformers first, then raw transformers."""
        print(f"[Embedder] Loading model: {self.model_name}")
        print("[Embedder] (First run will download ~80MB model - cached for future use)")

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.backend = "sentence_transformers"
                dim = self.model.get_sentence_embedding_dimension()
                print(f"[Embedder] Loaded via sentence-transformers | dim={dim}")
                return
            except Exception as e:
                print(f"[Embedder] sentence-transformers failed: {e}")

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.backend = "transformers"
                print(f"[Embedder] Loaded via transformers")
                return
            except Exception as e:
                print(f"[Embedder] transformers failed: {e}")

        raise RuntimeError(
            "No embedding backend available. "
            "Install with: pip install sentence-transformers"
        )

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to aggregate token embeddings into sentence embedding."""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed (code snippet, function, query, etc.)

        Returns:
            List of floats representing the embedding vector
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
        """
        Embed a list of texts in batches for efficiency.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.backend == "sentence_transformers":
            return self._embed_sentence_transformers(texts, batch_size)
        elif self.backend == "transformers":
            return self._embed_transformers(texts, batch_size)
        else:
            raise RuntimeError("No embedding backend loaded")

    def _embed_sentence_transformers(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Embed using sentence-transformers library."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def _embed_transformers(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Embed using raw HuggingFace transformers with mean pooling."""
        import torch
        import torch.nn.functional as F

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            if len(texts) > 50:
                print(f"[Embedder] Batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            )

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pooling(output, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self.backend == "sentence_transformers":
            return self.model.get_sentence_embedding_dimension()
        elif self.backend == "transformers":
            return self.model.config.hidden_size
        return 0

    def __repr__(self) -> str:
        return f"CodeEmbedder(model={self.model_name}, backend={self.backend}, dim={self.embedding_dim})"

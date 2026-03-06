"""
Tests for AI Codebase Search Engine
Tests use a lightweight mock embedder to avoid downloading models in CI.
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_parser import CodeParser
from search_engine import CodebaseSearchEngine, display_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_codebase(tmp_path):
    """Create a temporary codebase with sample files for testing."""

    # Python file with functions
    py_file = tmp_path / "auth.py"
    py_file.write_text('''"""Authentication module."""

import hashlib
import jwt


def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return check_database(username, hashed)


def generate_jwt_token(user_id: int, secret: str) -> str:
    """Generate a JWT token for a user."""
    payload = {"user_id": user_id, "exp": 3600}
    return jwt.encode(payload, secret, algorithm="HS256")


class AuthMiddleware:
    """Authentication middleware for request processing."""

    def __init__(self, secret_key: str):
        self.secret = secret_key

    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token."""
        return jwt.decode(token, self.secret, algorithms=["HS256"])
''')

    # JavaScript file
    js_file = tmp_path / "database.js"
    js_file.write_text('''// Database connection module

const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

async function connectDatabase() {
  const client = await pool.connect();
  return client;
}

async function queryDatabase(sql, params) {
  const result = await pool.query(sql, params);
  return result.rows;
}

function closeConnection() {
  pool.end();
}

module.exports = { connectDatabase, queryDatabase, closeConnection };
''')

    # Generic text file
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project\nThis is a test codebase.\n")

    return tmp_path


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns random embeddings without loading models."""
    embedder = MagicMock()

    def fake_embed(text):
        # Deterministic embedding based on text content hash
        np.random.seed(hash(text[:50]) % (2**31))
        vec = np.random.randn(384).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

    def fake_embed_batch(texts, batch_size=64):
        return [fake_embed(t) for t in texts]

    embedder.embed = fake_embed
    embedder.embed_batch = fake_embed_batch
    return embedder


# ---------------------------------------------------------------------------
# CodeParser Tests
# ---------------------------------------------------------------------------

class TestCodeParser:
    """Tests for the CodeParser module."""

    def setup_method(self):
        self.parser = CodeParser()

    def test_parse_python_file(self, temp_codebase):
        """Test that Python files are parsed into function/class chunks."""
        py_file = temp_codebase / "auth.py"
        chunks = self.parser.parse_file(py_file, temp_codebase)

        assert len(chunks) > 0, "Should produce at least one chunk"

        # Check we got function chunks
        function_chunks = [c for c in chunks if c.get("chunk_type") == "function"]
        assert len(function_chunks) >= 2, "Should extract at least 2 functions"

        # Check function names are captured
        func_names = {c["function"] for c in function_chunks if c.get("function")}
        assert "authenticate_user" in func_names
        assert "generate_jwt_token" in func_names

    def test_parse_python_class(self, temp_codebase):
        """Test that Python classes are extracted."""
        py_file = temp_codebase / "auth.py"
        chunks = self.parser.parse_file(py_file, temp_codebase)

        class_chunks = [c for c in chunks if c.get("chunk_type") == "class"]
        assert len(class_chunks) >= 1
        class_names = {c["class"] for c in class_chunks if c.get("class")}
        assert "AuthMiddleware" in class_names

    def test_parse_js_file(self, temp_codebase):
        """Test JS file parsing."""
        js_file = temp_codebase / "database.js"
        chunks = self.parser.parse_file(js_file, temp_codebase)

        assert len(chunks) > 0
        assert any("database.js" in c["file"] for c in chunks)

    def test_parse_markdown_file(self, temp_codebase):
        """Test generic chunking for markdown files."""
        md_file = temp_codebase / "README.md"
        chunks = self.parser.parse_file(md_file, temp_codebase)

        assert len(chunks) > 0
        assert any("README.md" in c["file"] for c in chunks)

    def test_chunk_metadata_fields(self, temp_codebase):
        """Test that all required metadata fields are present in chunks."""
        py_file = temp_codebase / "auth.py"
        chunks = self.parser.parse_file(py_file, temp_codebase)

        required_fields = {"text", "file", "function", "class", "chunk_type", "start_line", "end_line"}
        for chunk in chunks:
            for field in required_fields:
                assert field in chunk, f"Missing field '{field}' in chunk: {chunk}"

    def test_empty_file_returns_no_chunks(self, tmp_path):
        """Test that empty files are handled gracefully."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        chunks = self.parser.parse_file(empty_file, tmp_path)
        assert chunks == []

    def test_relative_paths_used(self, temp_codebase):
        """Test that chunk file paths are relative to base_path."""
        py_file = temp_codebase / "auth.py"
        chunks = self.parser.parse_file(py_file, temp_codebase)

        for chunk in chunks:
            assert not chunk["file"].startswith("/"), f"Path should be relative: {chunk['file']}"
            assert "auth.py" in chunk["file"]

    def test_get_supported_extensions(self):
        """Test supported extensions list."""
        exts = self.parser.get_supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".go" in exts
        assert ".rs" in exts

    def test_generic_chunk_large_file(self, tmp_path):
        """Test sliding window chunking for large files."""
        large_file = tmp_path / "large.sh"
        # Write 100 lines
        content = "\n".join([f"echo 'line {i}'" for i in range(100)])
        large_file.write_text(content)

        chunks = self.parser.parse_file(large_file, tmp_path)
        assert len(chunks) > 1, "Large file should produce multiple chunks"


# ---------------------------------------------------------------------------
# SearchEngine Tests
# ---------------------------------------------------------------------------

class TestCodebaseSearchEngine:
    """Tests for the CodebaseSearchEngine class."""

    def test_build_index(self, temp_codebase, mock_embedder):
        """Test that index is built correctly."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            engine.build_index()

        assert engine.index is not None
        assert len(engine.metadata) > 0
        assert engine.index.shape[0] == len(engine.metadata)

    def test_search_returns_results(self, temp_codebase, mock_embedder):
        """Test that search returns results with correct structure."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            engine.build_index()
            results = engine.search("authentication middleware", top_k=3)

        assert len(results) <= 3
        assert len(results) > 0

        for result in results:
            assert "file" in result
            assert "score" in result
            assert "text" in result
            assert 0.0 <= result["score"] <= 1.0

    def test_search_empty_index(self, temp_codebase, mock_embedder):
        """Test that searching an empty index returns no results."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            # Don't build index
            results = engine.search("some query")

        assert results == []

    def test_top_k_limits_results(self, temp_codebase, mock_embedder):
        """Test that top_k parameter correctly limits result count."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            engine.build_index()

            results_3 = engine.search("function", top_k=3)
            results_1 = engine.search("function", top_k=1)

        assert len(results_3) <= 3
        assert len(results_1) <= 1

    def test_results_sorted_by_score(self, temp_codebase, mock_embedder):
        """Test that results are sorted by descending score."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            engine.build_index()
            results = engine.search("authentication", top_k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    def test_index_caching(self, temp_codebase, mock_embedder):
        """Test that index is cached to disk and reloaded."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine1 = CodebaseSearchEngine(str(temp_codebase))
            engine1.build_index()
            index_shape = engine1.index.shape

            # Second build should use cache
            engine2 = CodebaseSearchEngine(str(temp_codebase))
            engine2.build_index()

        assert engine2.index is not None
        assert engine2.index.shape == index_shape

    def test_force_rebuild_ignores_cache(self, temp_codebase, mock_embedder):
        """Test that force_rebuild=True ignores cached index."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            engine.build_index()
            engine.build_index(force_rebuild=True)  # Should not raise

        assert engine.index is not None

    def test_empty_codebase(self, tmp_path, mock_embedder):
        """Test behavior with empty codebase (no supported files)."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(tmp_path))
            engine.build_index()

        assert engine.index is None
        assert engine.metadata == []

    def test_codebase_hash_changes_on_modification(self, temp_codebase, mock_embedder):
        """Test that hash changes when codebase is modified."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            hash1 = engine._get_codebase_hash()

            # Modify a file
            (temp_codebase / "auth.py").write_text("# modified")
            hash2 = engine._get_codebase_hash()

        assert hash1 != hash2


# ---------------------------------------------------------------------------
# Display / Output Tests
# ---------------------------------------------------------------------------

class TestDisplayResults:
    """Tests for the display_results function."""

    def test_display_no_results(self, capsys):
        """Test display with no results."""
        display_results([], "test query")
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_display_with_results(self, capsys):
        """Test display with sample results."""
        results = [
            {
                "score": 0.85,
                "file": "auth.py",
                "function": "authenticate_user",
                "class": None,
                "start_line": 10,
                "end_line": 20,
                "chunk_type": "function",
                "text": "def authenticate_user(username, password):\n    pass"
            }
        ]
        display_results(results, "authentication middleware")
        captured = capsys.readouterr()
        assert "authentication middleware" in captured.out
        assert "auth.py" in captured.out
        assert "authenticate_user" in captured.out
        assert "0.8500" in captured.out

    def test_display_long_snippet_truncated(self, capsys):
        """Test that long code snippets are truncated."""
        long_text = "\n".join([f"line {i}" for i in range(50)])
        results = [
            {
                "score": 0.7,
                "file": "test.py",
                "function": "my_func",
                "class": None,
                "start_line": 1,
                "end_line": 50,
                "chunk_type": "function",
                "text": long_text
            }
        ]
        display_results(results, "query")
        captured = capsys.readouterr()
        assert "more lines" in captured.out


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests that test the full search pipeline."""

    def test_end_to_end_search(self, temp_codebase, mock_embedder):
        """Test complete pipeline: index -> search -> results."""
        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            engine.build_index()
            results = engine.search("JWT token authentication", top_k=5)

        assert len(results) > 0

        # All results should reference files in the temp codebase
        for result in results:
            assert result["file"] in ["auth.py", "database.js", "README.md"]
            assert isinstance(result["score"], float)

    def test_output_json_structure(self, temp_codebase, mock_embedder, tmp_path):
        """Test that output JSON file has correct structure."""
        output_file = tmp_path / "results.json"

        with patch("search_engine.CodeEmbedder", return_value=mock_embedder):
            engine = CodebaseSearchEngine(str(temp_codebase))
            engine.build_index()
            results = engine.search("database connection", top_k=3)

        # Save results to JSON
        with open(output_file, "w") as f:
            json.dump(results, f)

        # Reload and verify
        with open(output_file) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        for item in loaded:
            assert "file" in item
            assert "score" in item
            assert "text" in item

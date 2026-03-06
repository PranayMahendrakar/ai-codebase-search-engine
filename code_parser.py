"""
Code Parser Module
Parses source code files into meaningful chunks (functions, classes, methods).
Supports Python AST parsing + generic line-based chunking for other languages.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# Max lines per chunk for generic files
CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


class CodeParser:
    """
    Parses code files into semantically meaningful chunks.
    - Python: Uses AST to extract functions, classes, methods with docstrings
    - Other languages: Uses regex patterns + sliding window chunking
    """

    def parse_file(self, file_path: Path, base_path: Path) -> List[Dict]:
        """
        Parse a source file into chunks with metadata.

        Args:
            file_path: Path to the source file
            base_path: Base path of the codebase (for relative path display)

        Returns:
            List of chunk dicts with text, metadata, and location info
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                return []

            relative_path = str(file_path.relative_to(base_path))

            if file_path.suffix == ".py":
                chunks = self._parse_python(content, relative_path)
            elif file_path.suffix in {".js", ".ts", ".jsx", ".tsx"}:
                chunks = self._parse_js_ts(content, relative_path)
            elif file_path.suffix in {".java", ".cs", ".cpp", ".c", ".h", ".hpp"}:
                chunks = self._parse_c_style(content, relative_path)
            elif file_path.suffix in {".go"}:
                chunks = self._parse_go(content, relative_path)
            elif file_path.suffix in {".rs"}:
                chunks = self._parse_rust(content, relative_path)
            else:
                chunks = self._generic_chunk(content, relative_path, file_path.suffix)

            # Always add a file-level chunk for overall context
            file_chunk = {
                "text": self._make_file_summary(content, relative_path),
                "file": relative_path,
                "function": None,
                "class": None,
                "chunk_type": "file_summary",
                "start_line": 1,
                "end_line": min(30, len(content.splitlines())),
                "language": file_path.suffix.lstrip(".")
            }
            chunks.insert(0, file_chunk)
            return chunks

        except Exception as e:
            # Return a minimal chunk even on parse error
            return [{
                "text": f"File: {file_path.name} (parse error: {e})",
                "file": str(file_path.relative_to(base_path)),
                "function": None,
                "class": None,
                "chunk_type": "error",
                "start_line": 1,
                "end_line": 1,
                "language": file_path.suffix.lstrip(".")
            }]

    def _make_file_summary(self, content: str, relative_path: str) -> str:
        """Create a file-level summary chunk."""
        lines = content.splitlines()
        header = "\n".join(lines[:30])
        return f"File: {relative_path}\n\n{header}"

    # -------------------------------------------------------------------------
    # Python Parser (AST-based)
    # -------------------------------------------------------------------------

    def _parse_python(self, content: str, relative_path: str) -> List[Dict]:
        """Parse Python using AST for accurate function/class extraction."""
        chunks = []
        lines = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._generic_chunk(content, relative_path, ".py")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_python_function(node, lines, relative_path)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._extract_python_class(node, lines, relative_path)
                if chunk:
                    chunks.append(chunk)

        # Also add line-based chunks for top-level code
        top_level_chunks = self._extract_top_level_code(tree, lines, relative_path)
        chunks.extend(top_level_chunks)

        return chunks

    def _extract_python_function(
        self, node: ast.FunctionDef, lines: List[str], relative_path: str
    ) -> Optional[Dict]:
        """Extract a Python function/method as a chunk."""
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, "end_lineno") else start + 20
        end = min(end, len(lines))

        code_lines = lines[start:end]
        code_text = "\n".join(code_lines)

        # Get docstring if present
        docstring = ast.get_docstring(node) or ""

        # Build enriched text for better semantic search
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"

        enriched = f"Function: {node.name}\nSignature: {signature}\n"
        if docstring:
            enriched += f"Description: {docstring}\n"
        enriched += f"\n{code_text}"

        # Detect parent class
        parent_class = None
        if hasattr(node, "_parent"):
            if isinstance(node._parent, ast.ClassDef):
                parent_class = node._parent.name

        return {
            "text": enriched,
            "file": relative_path,
            "function": node.name,
            "class": parent_class,
            "chunk_type": "function",
            "start_line": node.lineno,
            "end_line": end,
            "language": "python"
        }

    def _extract_python_class(
        self, node: ast.ClassDef, lines: List[str], relative_path: str
    ) -> Optional[Dict]:
        """Extract a Python class definition as a chunk."""
        start = node.lineno - 1
        # For classes, just include the header + docstring (not the full body)
        end = min(start + 25, len(lines))

        code_text = "\n".join(lines[start:end])
        docstring = ast.get_docstring(node) or ""

        bases = [ast.unparse(b) if hasattr(ast, "unparse") else "..." for b in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"

        enriched = f"Class: {node.name}\nDefinition: {signature}\n"
        if docstring:
            enriched += f"Description: {docstring}\n"
        enriched += f"\n{code_text}"

        return {
            "text": enriched,
            "file": relative_path,
            "function": None,
            "class": node.name,
            "chunk_type": "class",
            "start_line": node.lineno,
            "end_line": end,
            "language": "python"
        }

    def _extract_top_level_code(
        self, tree: ast.Module, lines: List[str], relative_path: str
    ) -> List[Dict]:
        """Extract top-level imports and module-level code."""
        chunks = []
        import_lines = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(lines[node.lineno - 1])

        if import_lines:
            chunks.append({
                "text": f"Imports in {relative_path}:\n" + "\n".join(import_lines),
                "file": relative_path,
                "function": None,
                "class": None,
                "chunk_type": "imports",
                "start_line": 1,
                "end_line": len(import_lines),
                "language": "python"
            })

        return chunks

    # -------------------------------------------------------------------------
    # JavaScript/TypeScript Parser (Regex-based)
    # -------------------------------------------------------------------------

    def _parse_js_ts(self, content: str, relative_path: str) -> List[Dict]:
        """Parse JS/TS files extracting functions and classes via regex."""
        chunks = []
        lines = content.splitlines()

        # Match function declarations and arrow functions
        func_patterns = [
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>",
            r"^\s*(\w+)\s*(?:=\s*)?(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>",
            r"^(?:export\s+)?class\s+(\w+)",
            r"^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{",  # methods
        ]

        i = 0
        while i < len(lines):
            for pattern in func_patterns:
                match = re.match(pattern, lines[i])
                if match:
                    name = match.group(1)
                    # Find the end of this block
                    end = self._find_block_end(lines, i)
                    chunk_lines = lines[i:end]
                    chunk_type = "class" if "class" in lines[i] else "function"

                    chunks.append({
                        "text": f"{'Class' if chunk_type == 'class' else 'Function'}: {name}\n\n" + "\n".join(chunk_lines[:50]),
                        "file": relative_path,
                        "function": name if chunk_type == "function" else None,
                        "class": name if chunk_type == "class" else None,
                        "chunk_type": chunk_type,
                        "start_line": i + 1,
                        "end_line": min(end, len(lines)),
                        "language": "javascript"
                    })
                    break
            i += 1

        # Add sliding window chunks as fallback
        if not chunks:
            chunks = self._generic_chunk(content, relative_path, ".js")

        return chunks

    # -------------------------------------------------------------------------
    # Generic C-style languages (Java, C++, C#)
    # -------------------------------------------------------------------------

    def _parse_c_style(self, content: str, relative_path: str) -> List[Dict]:
        """Parse C-style languages via regex."""
        chunks = []
        lines = content.splitlines()

        # Class/interface patterns
        class_pattern = re.compile(
            r"^(?:public|private|protected|internal)?\s*(?:abstract|sealed|static)?\s*(?:class|interface|struct|enum)\s+(\w+)"
        )
        # Method patterns
        method_pattern = re.compile(
            r"^\s*(?:public|private|protected|static|final|override|virtual|async)?\s*(?:\w+[\w<>\[\]]*\s+)+(\w+)\s*\([^;{]*\)\s*(?:throws\s+\w+)?\s*\{"
        )

        i = 0
        while i < len(lines):
            line = lines[i]
            cm = class_pattern.match(line)
            mm = method_pattern.match(line)

            if cm:
                name = cm.group(1)
                end = min(i + 30, len(lines))
                chunks.append({
                    "text": f"Class: {name}\n\n" + "\n".join(lines[i:end]),
                    "file": relative_path,
                    "function": None,
                    "class": name,
                    "chunk_type": "class",
                    "start_line": i + 1,
                    "end_line": end,
                    "language": "java"
                })
            elif mm:
                name = mm.group(1)
                end = self._find_block_end(lines, i)
                chunk_text = "\n".join(lines[i:min(end, i + 60)])
                chunks.append({
                    "text": f"Method: {name}\n\n{chunk_text}",
                    "file": relative_path,
                    "function": name,
                    "class": None,
                    "chunk_type": "function",
                    "start_line": i + 1,
                    "end_line": min(end, len(lines)),
                    "language": "java"
                })
            i += 1

        if not chunks:
            chunks = self._generic_chunk(content, relative_path, ".java")

        return chunks

    # -------------------------------------------------------------------------
    # Go Parser
    # -------------------------------------------------------------------------

    def _parse_go(self, content: str, relative_path: str) -> List[Dict]:
        """Parse Go source files."""
        chunks = []
        lines = content.splitlines()

        func_pattern = re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(")

        i = 0
        while i < len(lines):
            match = func_pattern.match(lines[i])
            if match:
                name = match.group(1)
                end = self._find_block_end(lines, i)
                chunk_text = "\n".join(lines[i:min(end, i + 60)])
                chunks.append({
                    "text": f"Function: {name}\n\n{chunk_text}",
                    "file": relative_path,
                    "function": name,
                    "class": None,
                    "chunk_type": "function",
                    "start_line": i + 1,
                    "end_line": min(end, len(lines)),
                    "language": "go"
                })
            i += 1

        if not chunks:
            chunks = self._generic_chunk(content, relative_path, ".go")

        return chunks

    # -------------------------------------------------------------------------
    # Rust Parser
    # -------------------------------------------------------------------------

    def _parse_rust(self, content: str, relative_path: str) -> List[Dict]:
        """Parse Rust source files."""
        chunks = []
        lines = content.splitlines()

        fn_pattern = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(")
        struct_pattern = re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)")

        i = 0
        while i < len(lines):
            line = lines[i]
            fn_match = fn_pattern.match(line)
            struct_match = struct_pattern.match(line)

            if fn_match:
                name = fn_match.group(1)
                end = self._find_block_end(lines, i)
                chunk_text = "\n".join(lines[i:min(end, i + 60)])
                chunks.append({
                    "text": f"Function: {name}\n\n{chunk_text}",
                    "file": relative_path,
                    "function": name,
                    "class": None,
                    "chunk_type": "function",
                    "start_line": i + 1,
                    "end_line": min(end, len(lines)),
                    "language": "rust"
                })
            elif struct_match:
                name = struct_match.group(1)
                end = min(i + 30, len(lines))
                chunks.append({
                    "text": f"Struct/Trait: {name}\n\n" + "\n".join(lines[i:end]),
                    "file": relative_path,
                    "function": None,
                    "class": name,
                    "chunk_type": "struct",
                    "start_line": i + 1,
                    "end_line": end,
                    "language": "rust"
                })
            i += 1

        if not chunks:
            chunks = self._generic_chunk(content, relative_path, ".rs")

        return chunks

    # -------------------------------------------------------------------------
    # Generic Chunker (sliding window)
    # -------------------------------------------------------------------------

    def _generic_chunk(self, content: str, relative_path: str, ext: str) -> List[Dict]:
        """
        Generic sliding window chunker for unsupported languages.
        Splits content into overlapping chunks of CHUNK_SIZE lines.
        """
        lines = content.splitlines()
        chunks = []
        lang = ext.lstrip(".")

        i = 0
        while i < len(lines):
            end = min(i + CHUNK_SIZE, len(lines))
            chunk_lines = lines[i:end]
            chunk_text = "\n".join(chunk_lines)

            if chunk_text.strip():
                chunks.append({
                    "text": f"File: {relative_path} (lines {i+1}-{end})\n\n{chunk_text}",
                    "file": relative_path,
                    "function": None,
                    "class": None,
                    "chunk_type": "chunk",
                    "start_line": i + 1,
                    "end_line": end,
                    "language": lang
                })

            i += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _find_block_end(self, lines: List[str], start: int) -> int:
        """Find the end of a code block by counting braces."""
        depth = 0
        found_open = False

        for i, line in enumerate(lines[start:], start):
            depth += line.count("{") - line.count("}")
            if "{" in line:
                found_open = True
            if found_open and depth <= 0:
                return i + 1

        return min(start + 60, len(lines))

    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cs", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".rb", ".php", ".swift", ".kt",
            ".sh", ".bash", ".sql", ".yaml", ".yml",
            ".html", ".css", ".md", ".json"
        ]

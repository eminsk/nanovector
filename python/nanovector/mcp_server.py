"""
Native Model Context Protocol (MCP) Server for NanoVector.
Provides sub-millisecond, zero-dependency episodic memory (remember, recall, stats)
backed by NanoVector's bare-metal C99 AVX2/NEON/FASM engine (.nvec single-file storage)
for Claude Desktop, Cursor, Windsurf, Antigravity, and any MCP client over JSON-RPC 2.0 stdio.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nanovector import Index, __version__, simd_backend

MCP_PROTOCOL_VERSION = "2024-11-05"


def embed_text(text: str, dim: int = 384) -> List[float]:
    """
    Zero-dependency multilingual (RU/EN) feature-hashing + character n-gram embedder.
    Produces L2-normalized float vectors of size `dim` in <0.1 ms with zero external models.
    """
    vec = [0.0] * dim
    cleaned = text.lower().strip()
    if not cleaned:
        vec[0] = 1.0
        return vec

    words = re.findall(r"[\w\-]+", cleaned, flags=re.UNICODE)
    features: List[tuple[str, float]] = []

    # Word unigrams (weight 2.0) & bigrams (weight 1.5)
    for i, w in enumerate(words):
        features.append((f"w:{w}", 2.0))
        if i + 1 < len(words):
            features.append((f"b:{w}_{words[i + 1]}", 1.5))
        # Character 3-grams and 4-grams within word for morphological robustness (Russian/English)
        padded = f"<{w}>"
        for n in (3, 4):
            for j in range(len(padded) - n + 1):
                features.append((f"c{n}:{padded[j:j + n]}", 0.75))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % dim
        sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 1e-12:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0
    return vec


MCP_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "nanovector_remember",
        "description": (
            "Store a fact, user preference, architectural decision, or code snippet into "
            "NanoVector's persistent single-file (.nvec) episodic memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique memory ID (e.g. 'pref_python_version', 'arch_db')"},
                "text": {"type": "string", "description": "Fact or content to remember"},
                "category": {"type": "string", "description": "Optional category tag (e.g. 'preference', 'code', 'architecture')"},
                "metadata": {"type": "object", "description": "Optional JSON metadata dictionary"},
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional explicit float embedding vector (auto-computed from text if omitted)",
                },
            },
            "required": ["id", "text"],
        },
    },
    {
        "name": "nanovector_recall",
        "description": (
            "Search NanoVector episodic memory using sub-millisecond SIMD similarity search "
            "(AVX2+FMA / ARM NEON / FASM) with optional metadata filtering."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "top_k": {"type": "integer", "default": 5, "description": "Maximum number of memories to recall"},
                "category": {"type": "string", "description": "Optional category filter"},
                "filter": {"type": "object", "description": "Optional NanoVector metadata filter ($eq, $in, $gte, etc.)"},
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional explicit query vector (auto-computed from query text if omitted)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "nanovector_stats",
        "description": "Inspect NanoVector index statistics, active hardware SIMD backend, and .nvec storage path.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class NanoVectorMCPServer:
    """Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server for NanoVector."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        dim: int = 384,
        metric: str = "cosine",
    ):
        default_path = str(Path.home() / ".nanovector" / "memory.nvec")
        self.db_path = db_path or os.environ.get("NANOVECTOR_DB", default_path)
        self.dim = int(os.environ.get("NANOVECTOR_DIM", str(dim)))
        self.metric = metric
        self.index = self._load_or_create()

    def _load_or_create(self) -> Index:
        p = Path(self.db_path)
        if p.exists():
            try:
                loaded = Index.load(str(p))
                self.dim = loaded.dim
                self.metric = loaded.metric
                return loaded
            except Exception:
                pass
        return Index(dim=self.dim, metric=self.metric, normalize=True)

    def _persist(self) -> None:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(p))

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC 2.0 MCP request."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        if req_id is None and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "nanovector-mcp",
                            "version": __version__,
                            "simdBackend": simd_backend(),
                        },
                    },
                }

            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SCHEMA}}

            if method == "tools/call":
                tool_name = params.get("name", "")
                args = params.get("arguments") or {}
                output = self._call_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(output, ensure_ascii=False, indent=2),
                            }
                        ],
                        "isError": False,
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "nanovector_remember":
            mem_id = str(args["id"])
            text = str(args["text"])
            vec = args.get("vector") or embed_text(text, dim=self.dim)
            meta: Dict[str, Any] = dict(args.get("metadata") or {})
            meta["text"] = text
            if args.get("category"):
                meta["category"] = str(args["category"])
            meta.setdefault("timestamp", int(time.time()))

            self.index.add(id=mem_id, vector=vec, metadata=meta)
            self._persist()
            return {
                "status": "stored",
                "id": mem_id,
                "total_memories": len(self.index),
                "db_path": self.db_path,
                "simd_backend": simd_backend(),
            }

        if name == "nanovector_recall":
            query_text = str(args.get("query", ""))
            vec = args.get("vector") or embed_text(query_text, dim=self.dim)
            top_k = int(args.get("top_k", 5))
            flt: Optional[Dict[str, Any]] = dict(args["filter"]) if args.get("filter") else None
            if args.get("category"):
                flt = flt or {}
                flt["category"] = str(args["category"])

            matches = self.index.search(query=vec, top_k=top_k, filter=flt)
            return {
                "query": query_text,
                "count": len(matches),
                "matches": [
                    {
                        "id": m.id,
                        "score": round(m.score, 4),
                        "metadata": m.meta,
                    }
                    for m in matches
                ],
            }

        if name == "nanovector_stats":
            return {
                "version": __version__,
                "simd_backend": simd_backend(),
                "count": len(self.index),
                "dim": self.index.dim,
                "metric": self.index.metric,
                "db_path": self.db_path,
            }

        raise ValueError(f"Unknown MCP tool: {name}")

    def run_stdio(self) -> None:
        """Run the NanoVector MCP JSON-RPC 2.0 server over standard input/output."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main_mcp(argv=None) -> int:
    """CLI entry point for nanovector-mcp."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="nanovector-mcp",
        description="Start NanoVector Model Context Protocol (MCP) Episodic Memory Server over stdio",
    )
    parser.add_argument("--db", default=None, help="Path to .nvec memory file (default: ~/.nanovector/memory.nvec)")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimensionality (default: 384)")
    parser.add_argument("--metric", default="cosine", help="Similarity metric: cosine, dot, l2 (default: cosine)")
    args = parser.parse_args(argv)

    server = NanoVectorMCPServer(db_path=args.db, dim=args.dim, metric=args.metric)
    server.run_stdio()
    return 0


if __name__ == "__main__":
    sys.exit(main_mcp())

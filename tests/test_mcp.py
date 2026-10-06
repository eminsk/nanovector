"""
Tests for NanoVector Model Context Protocol (MCP) JSON-RPC 2.0 Server.
"""

import json
from nanovector import NanoVectorMCPServer, embed_text


def test_embed_text_multilingual():
    v_ru = embed_text("Векторный поиск на чистом C99 и AVX2", dim=128)
    v_en = embed_text("Vector search in pure C99 and AVX2", dim=128)
    assert len(v_ru) == 128
    assert len(v_en) == 128
    assert abs(sum(x * x for x in v_ru) - 1.0) < 1e-5


def test_nanovector_mcp_server_lifecycle(tmp_path):
    db_file = tmp_path / "mcp_memory.nvec"
    server = NanoVectorMCPServer(db_path=str(db_file), dim=128)

    # 1. initialize
    init_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp["result"]["serverInfo"]["name"] == "nanovector-mcp"

    # 2. tools/list
    list_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = [t["name"] for t in list_resp["result"]["tools"]]
    assert "nanovector_remember" in tools
    assert "nanovector_recall" in tools
    assert "nanovector_stats" in tools

    # 3. remember two facts (Russian & English)
    r1 = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "nanovector_remember",
                "arguments": {
                    "id": "mem_1",
                    "text": "NanoVector использует AVX2 и FASM для быстрого векторного поиска",
                    "category": "architecture",
                },
            },
        }
    )
    assert r1["result"]["isError"] is False
    assert db_file.exists()

    r2 = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "nanovector_remember",
                "arguments": {
                    "id": "mem_2",
                    "text": "Пользователь предпочитает тёмную тему в редакторе",
                    "category": "preference",
                },
            },
        }
    )
    assert r2["result"]["isError"] is False

    # 4. recall from a fresh server instance reloading the .nvec file
    reloaded = NanoVectorMCPServer(db_path=str(db_file), dim=128)
    recall_resp = reloaded.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "nanovector_recall",
                "arguments": {
                    "query": "быстрый векторный поиск AVX2",
                    "top_k": 2,
                },
            },
        }
    )
    data = json.loads(recall_resp["result"]["content"][0]["text"])
    assert data["count"] == 2
    assert data["matches"][0]["id"] == "mem_1"

    # 5. stats
    stats_resp = reloaded.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nanovector_stats", "arguments": {}},
        }
    )
    stats = json.loads(stats_resp["result"]["content"][0]["text"])
    assert stats["count"] == 2
    assert stats["dim"] == 128

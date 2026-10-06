"""
CLI and MCP Server entry point for `python -m nanovector`.
"""

import sys
from nanovector.mcp_server import main_mcp

if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--mcp"]
    sys.exit(main_mcp(argv))

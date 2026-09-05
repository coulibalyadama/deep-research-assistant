# Deep Research Assistant

An AI-powered research assistant built on the Model Context Protocol (MCP) that orchestrates multiple services and tools. Implements Retrieval-Augmented Generation (RAG) with semantic search for comprehensive research workflows.

## Technologies & Tools

- **Model Context Protocol (MCP)**: Multi-server orchestration and tool integration
- **FastMCP**: MCP server implementation for research operations
- **LangChain & LangGraph**: LLM orchestration and agentic workflows
- **RAG (Retrieval-Augmented Generation)**: Semantic search and document retrieval
- **FAISS**: Vector database for efficient similarity search
- **HuggingFace Embeddings**: BGE-M3 for document embeddings
- **PyTorch & Transformers**: Deep learning and NLP models
- **Firecrawl API**: Web scraping and data extraction
- **Python 3.12+**

## Installation

**Prerequisites:** Python 3.12+, `uv` package manager

```bash
cd deep-research-assistant
uv sync
```

## Running the Project

Start the MCP research server:
```bash
uv run server.py
```

Run the agentic client:
```bash
uv run client.py
```

Or run the main entry point:
```bash
uv run main.py
```


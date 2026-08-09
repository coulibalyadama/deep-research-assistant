from fastmcp import FastMCP
from pathlib import Path
from typing import List

from langchain.vectors import FAISS
from langchain.embeddings import OpenAIEmbeddings
from lanchain.schema import Document

# Init MCP Server
mcp = FastMCP("Research Operations")

VECTOR_DB_ROOT = Path("vector_dbs/")

@mcp.tool()
def save_embeddings(documents: List[Document], path: str = "default") -> str:
    """
    Save documents to FAISS DB under vector_dbs/{path}/.
    Automatically creates the directory if it doesn't exist.
    """

    target_path = VECTOR_DB_ROOT / path
    target_path.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model="test-embedding-small",
                                  api_key="YOUR_OPENAI_API_KEY")

    documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]
    index_file = target_path / "index.faiss"

    if index_file.exists():
        vectorstore = FAISS.load_local(
            str(target_path), embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(documents)
        vectorstore.save_local(str(target_path))
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(str(target_path))

    return f"Saved {len(documents)} documents to {target_path}"

@mcp.tool()
def semantic_search(query: str, path: str = "default", k: int = 5) -> List[Document]:
    """
    Perform semantic search on the FAISS DB under vector_dbs/{path}/.
    Returns the top k most similar documents.
    """

    target_path = VECTOR_DB_ROOT / path
    index_file = target_path / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(f"No index found at {index_file}.") 

    embeddings = OpenAIEmbeddings(model="test-embedding-small",
                                  api_key="YOUR_OPENAI_API_KEY")

    vectorstore = FAISS.load_local(
        str(target_path), embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search(query, k=k)
    return [r.page_content for r in results]

@mcp.tool()
def available_prompts():
    """
    List all prompts available with the server. Give the exact name of the prompt to the user.
    """

    data = mcp.get_prompts()
    return data


@mcp.ressource("vector://list")
def list_vector_dbs() -> str:
    """
    Return a newline-separated list of available vector DBs (paths).
    Each represents a subfolder under vector-.dbs/.
    """

    if not VECTOR_DB_ROOT.exists():
        return "(No vector databases found.)"

    vector_dbs = [
        f"DB Name: {p.name}"
        for p in VECTOR_DB_ROOT.iterdir() 
        if (p / "index.faiss").exists()
    ]
    return "\n".join(vector_dbs)

@mcp.prompt()
def research_prompt(topic: str) -> str:
    """
    Provides a strutured deep research prompt on a given topic.
    """

    return f"""
    You are an AI researcher.
    Conduct a deep investigation into the topic:
    
    **{topic}**
    
    Provide:
    1. Definition and scope
    2. Current state of research
    3. Relevant papers and articles
    4. Key challenges and open questions
    
    Use advanced terminilogy and ensure scientific rigor.
    """

if __name__ == "__main__":
    mcp.run()
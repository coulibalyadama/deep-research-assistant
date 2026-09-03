from typing import List
from typing_extensions import TypedDict
from typing import Annotated
from langchain.prompts import ChataPromptTemplate, MessagesPlaceholder
from langchain_litellm import ChatLiteLLM
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.ressources import load_mcp_resources
from langchain_mcp_adapters.prompts import load_mcp_prompt
import asyncio
import json
from colorama import Fore, Stype, init

# Initialize colorama for colored output in the terminal
init(autoreset=True)

client = MultiServerMCPClient(
    {
        "first_server": {
            "command": "python",
            "args": ["server.py"],
            "transport": "stdio",
        },
        # config for firecrawl server
        "second_server": {
            "command": "npx",
            "args": ["-y", "firecrawl-mcp"],
            "transport": "stdio",
            "env": {
                "FIRECRAWL_API_KEY": "YOUR_FIRECRAWL_API_KEY",
            },
        },
    }
)


# Global state for resources
loaded_resources = {}

def extract_meta_command(message: str):

    if message.startswith("@resource:"):
        return "resource", message.split("@resource:")[1].strip().strip('"')

    elif message.startswith("@prompt:"):
        return "prompt", message.split("@prompt:")[1].strip().strip('"')

    elif message.startswith("@use_resource:"):
        parts = message.split(":", 1)[1].strip()

        space_index = parts.find(" ")

        if space_index != -1:
            resource_uri = parts[:space_index].strip()
            user_query = parts[space_index + 1:].strip()
            return "use_resource", (resource_uri, user_query)

        else:
            # No query provided after URI
            return "use_resource", (parts.strip(), "")
        
    return None, None

def parse_arguments(raw_input: str) -> dict:

    args = {}
    raw_input = raw_input.strip()

    # try JSON format first
    try:
        args = json.loads(raw_input)
        return args
    except json.JSONDecodeError:
        pass

    # Try key:value pairs separated by commas
    try:
        pairs = raw_input.split(",")
        for pair in pairs:
            if ":" in pair:
                key, value = pair.split(":", 1)
                args[key.strip()] = value.strip()
        return args
    except Exception:
        pass

    # Fallback to single key:value
    try:
        key, value = raw_input.split(":", 1)
        args[key.strip()] = value.strip()
        return args
    except Exception:
        pass

    # If all else fails, return an empty dict
    return args


from typing import List
from typing_extensions import TypedDict
from typing import Annotated
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_models import ChatLiteLLM
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.prompts import load_mcp_prompt

import asyncio
import json
from colorama import Fore, Style, init

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
            # "url": "https://mcp.firecrawl.dev/fc-api-key/sse",
            "url": "https://mcp.firecrawl.dev/v2/mcp",
            "transport": "streamable_http",
        }
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


# === Utility: Inject resource into message ===


def inject_resource_into_message(user_query: str,
                                 resource_uri: str) -> str:
    if resource_uri in loaded_resources:
        resource_content = loaded_resources[resource_uri]
        return f"""[USING RESOURCE: {resource_uri}]
{resource_content}

User query: {user_query}"""
    else:
        return f"[ERROR: Resource '{resource_uri}' not found. Available resources: {list(loaded_resources.keys())}]\n\nUser query: {user_query}"

# === Setup LangGraph workflow ===


async def create_graph(first_session, second_session):

    pipe = pipeline("text-generation", model="openai/gpt-oss-20B", torch_dtype="auto", device_map="auto", max_new_tokens=2048)
    llm = HuggingFacePipeline(pipeline=pipe)

    llm = ChatHuggingFace(llm=llm)
    # llm = ChatLiteLLM(
    #     model="gpt-4o", api_key="sk-...")

    # Load tools from both servers
    first_tools = await load_mcp_tools(first_session)
    second_tools = await load_mcp_tools(second_session)
    tools = first_tools + second_tools

    # Bind tools to LLM
    llm_with_tool = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a deep research assistant that uses tools to solve problems."),
        MessagesPlaceholder("messages")
    ])

    # State for LangGraph
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    # Main node
    def chat_node(state: State) -> State:
        chat_llm = prompt_template | llm_with_tool
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))

    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node", "__end__": END
    })
    graph_builder.add_edge("tool_node", "chat_node")

    return graph_builder.compile(checkpointer=MemorySaver()), tools

# === Main Entry Point ===


async def main():
    global loaded_resources
    config = {"configurable": {"thread_id": 1234}}

    async with client.session("first_server") as first_session, \
            client.session("second_server") as second_session:

        agent, tools = await create_graph(first_session, second_session)

        # Show available tools
        print(f"\n{Fore.CYAN}Available tools:{Style.RESET_ALL}")
        for tool in tools:
            print(f" - {tool.name}")

        all_resources = await load_mcp_resources(first_session)
        print(f"\n{Fore.CYAN}Available resources:{Style.RESET_ALL}")
        for resource in all_resources:
            print(f" - {resource.metadata['uri']}")

        print(f"\n{Fore.CYAN}Commands:{Style.RESET_ALL}")
        print(f" - @resource:<uri> - Load a resource")
        print(f" - @prompt:<name> - Load and execute a prompt")
        print(f" - @use_resource:<uri> <your_question> - Use a loaded resource in your question")
        print(f" - exit/quit - Exit the program")
        print(f"\n{Fore.CYAN}Argument formats:{Style.RESET_ALL}")
        print(f" - Simple: key:value")
        print(f" - Multiple: key1:value1, key2:value2")
        print(f" - JSON: {{\"key1\": \"value1\", \"key2\": \"value2\"}}")

        while True:
            message = input(f"\n{Fore.GREEN}User:{Style.RESET_ALL} ").strip()
            if message.lower() == "exit" or message.lower() == "quit":
                break

            # === Handle @prompt:, @resource:, or @use_resource: ===
            cmd_type, value = extract_meta_command(message)

            if cmd_type == "use_resource":
                resource_uri, user_query = value

                if not user_query:
                    print(
                        f"{Fore.RED}Please provide a query after the resource URI.{Style.RESET_ALL}")
                    print(
                        f"{Fore.YELLOW}Usage: @use_resource:<uri> <your_question>{Style.RESET_ALL}")
                    continue

                if resource_uri not in loaded_resources:
                    print(
                        f"{Fore.RED}Resource '{resource_uri}' has not been loaded yet.{Style.RESET_ALL}")
                    print(
                        f"{Fore.YELLOW}Please load it first using: @resource:{resource_uri}{Style.RESET_ALL}")
                    continue

                # Inject resource into the message
                enhanced_message = inject_resource_into_message(
                    user_query, resource_uri)
                try:
                    response = await agent.ainvoke({
                        "messages": enhanced_message
                    }, config=config)
                    print(
                        f"\n{Fore.BLUE}AI:{Style.RESET_ALL} {response['messages'][-1].content}")
                except Exception as e:
                    print(f"{Fore.RED}Agent error: {e}{Style.RESET_ALL}")
                continue
            elif cmd_type == "prompt":
                try:
                    # Try loading without arguments first
                    prompt = await load_mcp_prompt(first_session, value)

                    if prompt:
                        try:
                            # Take the first message or combine all messages
                            if len(prompt) == 1:
                                prompt_message = prompt[0].content
                            else:
                                # Combine all messages
                                prompt_message = "\n".join(
                                    [msg.content for msg in prompt])
                            print(
                                f"{Fore.YELLOW}Loaded and executing prompt: {value}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}Prompt content: {prompt_message[:200]}...{Style.RESET_ALL}" if len(
                                prompt_message) > 200 else f"{Fore.CYAN}Prompt content: {prompt_message}{Style.RESET_ALL}")
                        except Exception as e:
                            print(
                                f"{Fore.RED}Failed to load prompt: {e}{Style.RESET_ALL}")
                            continue

                        try:
                            response = await agent.ainvoke({
                                "messages": prompt_message
                            }, config=config)
                            print(
                                f"\n{Fore.BLUE}=== RESEARCH RESULTS ==={Style.RESET_ALL}")
                            print(
                                f"{Fore.BLUE}AI:{Style.RESET_ALL} {response['messages'][-1].content}")
                            print(
                                f"{Fore.BLUE}=== END RESEARCH RESULTS ==={Style.RESET_ALL}")
                        except Exception as e:
                            print(f"{Fore.RED}Agent error: {e}{Style.RESET_ALL}")
                    else:
                        print(
                            f"{Fore.RED}No prompt messages found{Style.RESET_ALL}")

                except Exception as e:
                    print(
                        f"{Fore.YELLOW}Prompt '{value}' requires arguments. Error: {e}{Style.RESET_ALL}")

                    # Show available argument formats
                    print(
                        f"{Fore.CYAN}Enter arguments (formats supported):{Style.RESET_ALL}")
                    print(f"  Simple: key:value")
                    print(f"  Multiple: key1:value1, key2:value2")
                    print(
                        f"  JSON: {{\"key1\": \"value1\", \"key2\": \"value2\"}}")

                    raw_input_str = input(
                        f"{Fore.CYAN}Arguments: {Style.RESET_ALL}").strip()

                    # Parse arguments using the improved parser
                    args = parse_arguments(raw_input_str)

                    if not args:
                        print(
                            f"{Fore.RED}Failed to parse arguments. Please check the format.{Style.RESET_ALL}")
                        continue

                    try:
                        prompt = await load_mcp_prompt(
                            first_session, value, arguments=args)
                        if prompt:
                            if len(prompt) == 1:
                                prompt_message = prompt[0].content
                            else:
                                prompt_message = "\n".join(
                                    [msg.content for msg in prompt])
                            print(
                                f"{Fore.YELLOW}Loaded and executing prompt: {value} with arguments: {args}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}Prompt content: {prompt_message[:500]}...{Style.RESET_ALL}" if len(
                                prompt_message) > 500 else f"{Fore.CYAN}Prompt content: {prompt_message}{Style.RESET_ALL}")

                            try:
                                response = await agent.ainvoke({
                                    "messages": prompt_message
                                }, config=config)
                                print(
                                    f"\n{Fore.BLUE}=== RESEARCH RESULTS ==={Style.RESET_ALL}")
                                print(
                                    f"{Fore.BLUE}AI:{Style.RESET_ALL} {response['messages'][-1].content}")
                                print(
                                    f"{Fore.BLUE}=== END RESEARCH RESULTS ==={Style.RESET_ALL}")
                            except Exception as e:
                                print(
                                    f"{Fore.RED}Agent error: {e}{Style.RESET_ALL}")
                        else:
                            print(
                                f"{Fore.RED}No prompt messages found{Style.RESET_ALL}")
                    except Exception as ex:
                        print(
                            f"{Fore.RED}Failed to load prompt with arguments: {ex}{Style.RESET_ALL}")
                        print(
                            f"{Fore.YELLOW}Arguments provided: {args}{Style.RESET_ALL}")
                continue

            elif cmd_type == "resource":
                try:
                    # Remove old copy if exists
                    if value in loaded_resources:
                        print(
                            f"{Fore.YELLOW}Updating existing resource: {value}{Style.RESET_ALL}")
                        del loaded_resources[value]

                    blob = await load_mcp_resources(first_session, uris=[value])
                    # print(blob[0].as_string())
                    if blob:
                        loaded_resources[value] = blob[0].as_string()
                        print(
                            f"{Fore.GREEN}Loaded resource: {value}{Style.RESET_ALL}")
                    else:
                        print(
                            f"{Fore.RED}No resource found with URI: {value}{Style.RESET_ALL}")
                except Exception as e:
                    print(
                        f"{Fore.RED}Failed to load resource: {e}{Style.RESET_ALL}")
                continue
            else:
                try:
                    response = await agent.ainvoke({
                        "messages": message
                    }, config=config)
                    print(
                        f"\n{Fore.BLUE}AI:{Style.RESET_ALL} {response['messages'][-1].content}")
                except Exception as e:
                    print(f"{Fore.RED}Agent error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())




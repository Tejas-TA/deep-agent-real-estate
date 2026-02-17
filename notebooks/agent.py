"""
Orlando Real Estate Agent
LangGraph ReAct agent connecting to the MCP server via stdio.

Usage:
    python agent.py                                            # interactive loop
    python agent.py --query "What size tree requires permit?"  # single query
    python agent.py --query "Walkability at 28.5383, -81.3792" --quiet
"""

import asyncio
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================

MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

MCP_CONFIG = {
    "orlando_realestate": {
        "command": "python",
        "args": [MCP_SERVER_PATH],
        "transport": "stdio",
        "env": {**os.environ},
    }
}

LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an expert Orlando real estate AI assistant with access to 5 specialized tools:

1. zoning_law_query           - Answer questions about Orlando municipal zoning codes
2. property_damage_assessment - Analyze property damage images or search similar cases
3. predict_fair_market_value  - Predict fair market value using XGBoost model
4. assess_walkability         - Score walkability using live OpenStreetMap data
5. orlando_market_expert      - Query fine-tuned AI for market trends and neighborhood insights

Guidelines:
- Use the most relevant tool(s) for each question
- For full property evaluations, combine multiple tools (FMV + walkability + market expert)
- Format dollar amounts with commas, scores as X/100
- If a tool errors, explain what went wrong and suggest alternatives
- Be concise but thorough
"""


# ============================================================================
# HELPERS
# ============================================================================

def _print_node(step: int, node_name: str, state: dict) -> str:
    """Print a graph node's state and return the AI response text if present."""
    print(f"\n📍 Node [{step}]: {node_name}")
    print("─" * 40)
    final = ""
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc["args"], indent=2)
                    print(f"  🔨 Tool call : {tc['name']}")
                    print(f"     Args      : {args_str[:300]}")
            elif msg.content:
                print(f"  🤖 AI: {msg.content[:400]}")
                final = msg.content
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content)[:300].replace("\n", " ")
            print(f"  📦 Tool result [{msg.name}]: {preview}...")
        elif isinstance(msg, HumanMessage):
            print(f"  👤 Human: {str(msg.content)[:200]}")
    return final


async def _build_agent():
    """Initialise MCP client, load tools, return (agent, tools)."""
    mcp_client = MultiServerMCPClient(MCP_CONFIG)
    tools = await mcp_client.get_tools()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    agent = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    return agent, tools


# ============================================================================
# SINGLE-QUERY RUNNER
# ============================================================================

async def run_agent(query: str, verbose: bool = True) -> str:
    """Run one query, stream graph node states, return final answer."""
    agent, tools = await _build_agent()

    if verbose:
        print(f"\n🔧 Tools loaded: {[t.name for t in tools]}")
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

    final_response = ""
    step = 0

    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="updates",
    ):
        step += 1
        for node_name, state in chunk.items():
            if verbose:
                answer = _print_node(step, node_name, state)
                if answer:
                    final_response = answer
            else:
                for msg in state.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        final_response = msg.content

    if verbose:
        print("\n" + "=" * 60)
        print("✅ Final Answer:")
        print("=" * 60)
        print(final_response)

    return final_response


# ============================================================================
# INTERACTIVE LOOP
# ============================================================================

async def interactive_loop():
    """REPL session — maintains conversation history across turns."""
    print("\n🏠 Orlando Real Estate AI Agent")
    print("Type your question or 'quit' to exit.\n")

    agent, tools = await _build_agent()
    print(f"✅ {len(tools)} tools ready: {[t.name for t in tools]}\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))

        print()
        step = 0
        final_answer = ""

        async for chunk in agent.astream(
            {"messages": messages},
            stream_mode="updates",
        ):
            step += 1
            for node_name, state in chunk.items():
                answer = _print_node(step, node_name, state)
                if answer:
                    final_answer = answer

        print(f"\nAgent: {final_answer}\n")
        messages.append(AIMessage(content=final_answer))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orlando Real Estate AI Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query (skips interactive mode)")
    parser.add_argument("--quiet", action="store_true", help="Suppress graph node trace output")
    args = parser.parse_args()

    if args.query:
        asyncio.run(run_agent(args.query, verbose=not args.quiet))
    else:
        asyncio.run(interactive_loop())

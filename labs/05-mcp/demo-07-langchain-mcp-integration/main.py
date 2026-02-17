"""
Demo 07: LangChain MCP Integration

Integrate MCP tools with LangChain agents:
- Convert FastMCP tools to LangChain tools
- Build agents that use MCP servers
- Framework interoperability
- Multi-tool orchestration

Key Concepts:
- FastMCP + LangChain integration
- Tool wrapping patterns
- Agent-based workflows
- Cross-framework compatibility
"""

import asyncio
import os
from typing import Any, Optional, Type
from dotenv import load_dotenv

# LangChain imports
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

# FastMCP import
from fastmcp import FastMCP

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
    print("⚠️  WARNING: OPENAI_API_KEY not configured")
    print("   This demo requires an OpenAI API key.")
    print("   1. Get key from https://platform.openai.com/")
    print("   2. Copy .env.example to .env")
    print("   3. Add your key to .env")
    print()
    exit(1)

print("=" * 70)
print("MCP DEMO 07: LANGCHAIN INTEGRATION")
print("=" * 70)
print(f"✓ OpenAI Model: {OPENAI_MODEL}")
print()

# ============================================================================
# FASTMCP SERVER (Calculator tools)
# ============================================================================

mcp = FastMCP("Calculator Server")


@mcp.tool()
def add(a: float, b: float) -> dict:
    """Add two numbers together."""
    result = a + b
    return {"result": result, "expression": f"{a} + {b} = {result}"}


@mcp.tool()
def subtract(a: float, b: float) -> dict:
    """Subtract b from a."""
    result = a - b
    return {"result": result, "expression": f"{a} - {b} = {result}"}


@mcp.tool()
def multiply(a: float, b: float) -> dict:
    """Multiply two numbers."""
    result = a * b
    return {"result": result, "expression": f"{a} × {b} = {result}"}


@mcp.tool()
def divide(a: float, b: float) -> dict:
    """Divide a by b."""
    if b == 0:
        return {"error": "Cannot divide by zero"}
    result = a / b
    return {"result": result, "expression": f"{a} ÷ {b} = {result}"}


# ============================================================================
# FASTMCP TO LANGCHAIN ADAPTER
# ============================================================================

class MCPTool(BaseTool):
    """
    LangChain tool wrapper for FastMCP tools.
    
    This adapter allows FastMCP tools to be used as LangChain tools,
    enabling integration with LangChain agents and chains.
    """
    
    name: str
    description: str
    mcp_tool_func: Any
    
    def __init__(self, mcp_tool_func: Any):
        """
        Initialize MCPTool from FastMCP tool function.
        
        Args:
            mcp_tool_func: The FastMCP tool function
        """
        super().__init__(
            name=mcp_tool_func.__name__,
            description=mcp_tool_func.__doc__ or f"FastMCP tool: {mcp_tool_func.__name__}",
            mcp_tool_func=mcp_tool_func
        )
    
    def _run(self, **kwargs) -> str:
        """
        Synchronous execution (required by LangChain).
        """
        try:
            result = self.mcp_tool_func(**kwargs)
            # Extract result from dict if present
            if isinstance(result, dict):
                if "error" in result:
                    return f"Error: {result['error']}"
                if "result" in result:
                    return str(result["result"])
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution."""
        return self._run(**kwargs)


def create_langchain_tools_from_fastmcp(mcp_instance: FastMCP) -> list[MCPTool]:
    """
    Convert all FastMCP tools to LangChain tools.
    
    Args:
        mcp_instance: FastMCP instance
        
    Returns:
        List of LangChain-compatible tools
    """
    langchain_tools = []
    
    # Access FastMCP's tool registry
    if hasattr(mcp_instance, '_tools'):
        for tool_func in mcp_instance._tools:
            lc_tool = MCPTool(mcp_tool_func=tool_func)
            langchain_tools.append(lc_tool)
    
    return langchain_tools


# ============================================================================
# LANGCHAIN AGENT
# ============================================================================

def create_agent_with_mcp_tools() -> AgentExecutor:
    """
    Create a LangChain agent with FastMCP tools.
    
    Returns:
        Configured agent executor
    """
    # Convert FastMCP tools to LangChain tools
    tools = create_langchain_tools_from_fastmcp(mcp)
    
    print(f"✓ Loaded {len(tools)} FastMCP tools into LangChain")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description}")
    print()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to calculator tools.
        
When asked to perform calculations:
1. Break down complex calculations into steps
2. Use the appropriate tools (add, subtract, multiply, divide)
3. Show your work
4. Provide the final answer clearly

Available tools:
- add(a, b): Add two numbers
- subtract(a, b): Subtract b from a
- multiply(a, b): Multiply two numbers
- divide(a, b): Divide a by b
"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


# ============================================================================
# DEMOS
# ============================================================================

async def demo_langchain_mcp():
    """Demonstrate LangChain agent using FastMCP tools."""
    
    print("=" * 70)
    print("LANGCHAIN + FASTMCP: Agent Demonstrations")
    print("=" * 70)
    print()
    
    # Create agent
    agent = create_agent_with_mcp_tools()
    
    # Test cases
    test_queries = [
        "What is 15 plus 27?",
        "Calculate (100 - 25) multiplied by 3", 
        "What is 144 divided by 12, then add 50?",
        "Solve: ((15 + 25) * 2) / 4"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'=' * 70}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 70}")
        print()
        
        try:
            result = agent.invoke({"input": query})
            print()
            print(f"✓ Final Answer: {result['output']}")
            print()
        except Exception as e:
            print(f"✗ Error: {e}")
            print()
    
    print("=" * 70)


async def main():
    """Run the LangChain+FastMCP integration demo."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WELCOME TO MCP DEMO 07" + " " * 31 + "║")
    print("║" + " " * 14 + "LangChain MCP Integration" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📚 This demo demonstrates:")
    print("   • Converting FastMCP tools to LangChain tools")
    print("   • Building agents with MCP tool access")
    print("   • Framework interoperability")
    print("   • Multi-tool orchestration by LLM")
    print("   • Real-world agent workflows")
    print()
    
    # Run demo
    await demo_langchain_mcp()
    
    print("=" * 70)
    print("💡 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("✓ FastMCP tools can be used in any framework via adapters")
    print("✓ LangChain agents can orchestrate multiple FastMCP tools")
    print("✓ Write tools once (FastMCP), use everywhere (LangChain, etc.)")
    print("✓ Standardization reduces integration complexity")
    print("✓ Agents decide which tools to use and in what order")
    print()
    
    print("=" * 70)
    print("🔍 HOW IT WORKS")
    print("=" * 70)
    print()
    print("1. FastMCP Server: Provides calculator tools")
    print("2. MCPTool Wrapper: Converts to LangChain format")
    print("3. LangChain Agent: Orchestrates tool usage")
    print("4. OpenAI LLM: Decides which tools to call")
    print("5. Tools Execute: FastMCP functions run")
    print("6. Results: Agent synthesizes final answer")
    print()
    
    print("=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Add more FastMCP servers:")
    print("   • Weather tools for real-time data")
    print("   • Database tools for data access")
    print("   • Filesystem tools for file operations")
    print()
    
    print("2. Explore advanced patterns:")
    print("   • Multi-server coordination")
    print("   • LangGraph for complex workflows")
    print("   • Streaming responses")
    print()
    
    print("💡 Exercise: Integrate Demo 03 weather tools")
    print("   Add weather.py server and use both calculators + weather")


if __name__ == "__main__":
    asyncio.run(main())

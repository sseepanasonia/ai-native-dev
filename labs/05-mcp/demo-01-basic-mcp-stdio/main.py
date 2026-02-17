"""
Demo 01: Basic MCP Server with stdio Transport

This is the foundational MCP demo that introduces:
- What is MCP (Model Context Protocol)
- Creating a simple MCP server with FastMCP
- Defining your first tool
- Using stdio transport (standard input/output)
- How MCP tools can be used by AI assistants

Key Concepts:
- MCP server setup with FastMCP
- Tool definition with @mcp.tool() decorator
- Stdio transport for local communication
- Tool discovery and execution
"""

from fastmcp import FastMCP

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

# Create an MCP server instance
# This server will be discoverable by MCP clients (like Claude Desktop)
mcp = FastMCP("demo-server")

print("=" * 70)
print("MCP DEMO 01: BASIC MCP SERVER")
print("=" * 70)
print()

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================


@mcp.tool()
def greet(name: str) -> str:
    """
    Greet someone by name.
    
    This is a simple tool that demonstrates the basics of MCP tool creation.
    
    Args:
        name: The name of the person to greet
        
    Returns:
        A friendly greeting message
    """
    print(f"[Server] Executing: greet(name='{name}')")
    greeting = f"Hello, {name}! Welcome to MCP (Model Context Protocol)!"
    print(f"[Server] Returning: {greeting}")
    return greeting


@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about this MCP server.
    
    Returns information about the server's capabilities and purpose.
    
    Returns:
        Dictionary containing server information
    """
    print("[Server] Executing: get_server_info()")
    info = {
        "name": "demo-server",
        "version": "1.0.0",
        "description": "Basic MCP server demonstrating stdio transport",
        "transport": "stdio",
        "tools_count": 2,
        "purpose": "Educational demo for learning MCP basics"
    }
    print(f"[Server] Returning: {info}")
    return info


# ============================================================================
# HELPER FUNCTIONS FOR DEMO (not MCP tools)
# ============================================================================

def _demo_greet(name: str) -> str:
    """Demo version of greet for testing."""
    greeting = f"Hello, {name}! Welcome to MCP (Model Context Protocol)!"
    return greeting


def _demo_get_server_info() -> dict:
    """Demo version of get_server_info for testing."""
    return {
        "name": "demo-server",
        "version": "1.0.0",
        "description": "Basic MCP server demonstrating stdio transport",
        "transport": "stdio",
        "tools_count": 2,
        "purpose": "Educational demo for learning MCP basics"
    }


# ============================================================================
# MAIN DEMO
# ============================================================================

def demo_introduction():
    """Display introduction and educational content."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WELCOME TO MCP DEMO 01" + " " * 31 + "║")
    print("║" + " " * 17 + "Basic MCP Server" + " " * 35 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("=" * 70)
    print("📚 WHAT IS MCP?")
    print("=" * 70)
    print()
    print("MCP (Model Context Protocol) is an open protocol that enables")
    print("seamless integration between LLM applications and external data sources.")
    print()
    print("Think of it as a universal connector that allows AI assistants")
    print("to access tools, databases, APIs, and services in a standardized way.")
    print()
    
    print("=" * 70)
    print("🎯 KEY CONCEPTS")
    print("=" * 70)
    print()
    print("1. SERVER: Exposes tools and resources (this demo)")
    print("2. CLIENT: Uses the tools (like Claude Desktop, or custom clients)")
    print("3. TRANSPORT: Communication method (stdio, HTTP, SSE)")
    print("4. TOOL: A function that the AI can call")
    print("5. PROTOCOL: Standardized way to discover and use tools")
    print()
    
    print("=" * 70)
    print("🔧 THIS DEMO SERVER")
    print("=" * 70)
    print()
    print("✓ Server Name: demo-server")
    print("✓ Transport: stdio (standard input/output)")
    print("✓ Tools: 2 simple demonstration tools")
    print("✓ Framework: FastMCP (simplified MCP server creation)")
    print()
    
    print("=" * 70)
    print("📋 AVAILABLE TOOLS")
    print("=" * 70)
    print()
    print("1. greet(name: str) -> str")
    print("   • Greets someone by name")
    print("   • Demonstrates basic tool execution")
    print()
    print("2. get_server_info() -> dict")
    print("   • Returns information about this server")
    print("   • Demonstrates returning structured data")
    print()
    
    print("=" * 70)
    print("🧪 DEMONSTRATION")
    print("=" * 70)
    print()
    print("Let's test our tools manually:")
    print()
    
    # Test greet tool
    print("Test 1: greet('Alice')")
    print("-" * 70)
    result1 = _demo_greet("Alice")
    print(f"✓ Result: {result1}")
    print()
    
    # Test get_server_info tool
    print("Test 2: get_server_info()")
    print("-" * 70)
    result2 = _demo_get_server_info()
    print("✓ Result:")
    for key, value in result2.items():
        print(f"  • {key}: {value}")
    print()
    
    print("=" * 70)
    print("💡 HOW IT WORKS")
    print("=" * 70)
    print()
    print("STEP 1: Define Your Tool")
    print("  @mcp.tool()")
    print("  def greet(name: str) -> str:")
    print('      """Greet someone by name."""')
    print('      return f"Hello, {name}!"')
    print()
    print("STEP 2: Run the Server")
    print("  mcp.run()")
    print()
    print("STEP 3: Client Discovers Tools")
    print("  • Client connects via stdio")
    print("  • Client asks: 'What tools do you have?'")
    print("  • Server responds with tool list and schemas")
    print()
    print("STEP 4: Client Uses Tools")
    print("  • Client sends: 'Call greet with name=Alice'")
    print("  • Server executes and returns result")
    print()
    
    print("=" * 70)
    print("🔐 STDIO TRANSPORT")
    print("=" * 70)
    print()
    print("stdio = Standard Input/Output")
    print()
    print("✓ Best for: Local tools and desktop applications")
    print("✓ Security: Process-level isolation")
    print("✓ Communication: JSON-RPC over stdin/stdout")
    print("✓ Use case: Claude Desktop, local AI assistants")
    print()
    print("Other transports:")
    print("  • HTTP: For web services (see demo-06)")
    print("  • SSE: For streaming updates")
    print()
    
    print("=" * 70)
    print("🚀 USING THIS SERVER")
    print("=" * 70)
    print()
    print("Option 1: Claude Desktop Integration")
    print("  1. Add server to Claude Desktop config")
    print("  2. Claude automatically discovers tools")
    print("  3. Ask Claude to use the greet tool")
    print()
    print("Option 2: Custom Client")
    print("  1. Create MCP client")
    print("  2. Connect to this server via stdio")
    print("  3. Call tools programmatically")
    print()
    print("See README.md for detailed setup instructions")
    print()
    
    print("=" * 70)
    print("💡 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("✓ FastMCP makes MCP server creation simple")
    print("✓ Use @mcp.tool() decorator to define tools")
    print("✓ Type hints automatically create JSON Schema")
    print("✓ Stdio transport enables local AI integration")
    print("✓ Tools are automatically discoverable by clients")
    print()
    
    print("=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Try demo-02: Multiple tools with validation")
    print("2. Try demo-03: External API integration")
    print("3. Try demo-04: Filesystem operations")
    print("4. Integrate with Claude Desktop (see README.md)")
    print()
    print("💡 Exercise: Add a new tool called 'calculate(a, b, operation)'")
    print("   that performs basic math operations")
    print()
    
    print("=" * 70)
    print("✨ Demo Complete!")
    print("=" * 70)
    print()


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # Run as MCP server (for production use with Claude Desktop)
        print("✓ Starting MCP server...")
        print("✓ Server: demo-server")
        print("✓ Transport: stdio")
        print("✓ Ready for client connections")
        print()
        mcp.run()
    else:
        # Run demo mode (educational)
        demo_introduction()
        print()
        print("=" * 70)
        print("🔧 TO RUN AS SERVER")
        print("=" * 70)
        print()
        print("  uv run python main.py --server")
        print()
        print("This will start the server and wait for client connections.")
        print()

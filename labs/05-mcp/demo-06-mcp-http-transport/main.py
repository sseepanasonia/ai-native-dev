"""
Demo 06: MCP HTTP Transport

MCP server using HTTP/REST API instead of stdio:
- FastAPI web server
- RESTful endpoints for MCP operations
- CORS support for web clients
- Health checks and monitoring
- OpenAPI documentation

Key Concepts:
- HTTP vs stdio transport
- RESTful API design
- Web-based tool calling
- FastMCP + FastAPI integration
"""

from typing import Any, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastmcp import FastMCP
import uvicorn
import inspect

# ============================================================================
# CONFIGURATION
# ============================================================================

# Create FastMCP server
mcp = FastMCP("HTTP Server")

# Create FastAPI app
app = FastAPI(
    title="MCP HTTP Server",
    description="Model Context Protocol server using HTTP transport",
    version="1.0.0"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("=" * 70)
print("MCP DEMO 06: HTTP TRANSPORT")
print("=" * 70)
print()

# ============================================================================
# MCP TOOLS (Using FastMCP)
# ============================================================================

@mcp.tool()
def greet(name: str, greeting: str = "Hello") -> dict:
    """
    Greet someone with a customizable greeting.
    
    Args:
        name: Name of person to greet
        greeting: Greeting word (default: "Hello")
    """
    print(f"[Tool] greet called: name={name}, greeting={greeting}")
    return {
        "message": f"{greeting}, {name}! Welcome to MCP over HTTP!",
        "name": name,
        "greeting": greeting
    }


@mcp.tool()
def add(a: float, b: float) -> dict:
    """Add two numbers together."""
    print(f"[Tool] add called: {a} + {b}")
    result = a + b
    return {
        "result": result,
        "expression": f"{a} + {b} = {result}"
    }


@mcp.tool()
def subtract(a: float, b: float) -> dict:
    """Subtract b from a."""
    print(f"[Tool] subtract called: {a} - {b}")
    result = a - b
    return {
        "result": result,
        "expression": f"{a} - {b} = {result}"
    }


@mcp.tool()
def multiply(a: float, b: float) -> dict:
    """Multiply two numbers."""
    print(f"[Tool] multiply called: {a} × {b}")
    result = a * b
    return {
        "result": result,
        "expression": f"{a} × {b} = {result}"
    }


@mcp.tool()
def divide(a: float, b: float) -> dict:
    """Divide a by b."""
    print(f"[Tool] divide called: {a} ÷ {b}")
    if b == 0:
        return {
            "error": "Cannot divide by zero",
            "expression": f"{a} ÷ {b} = undefined"
        }
    result = a / b
    return {
        "result": result,
        "expression": f"{a} ÷ {b} = {result}"
    }


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ToolCallRequest(BaseModel):
    """Request model for tool calls."""
    name: str
    arguments: dict = {}


class ToolInfo(BaseModel):
    """Tool information model."""
    name: str
    description: str
    parameters: Dict[str, Any]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_tool_function(tool_name: str):
    """Get the actual function for a tool by name."""
    # Access FastMCP's internal tool registry
    if hasattr(mcp, '_tools'):
        for tool in mcp._tools:
            if tool.__name__ == tool_name:
                return tool
    return None


def get_tool_info(func) -> dict:
    """Extract tool information from a function."""
    sig = inspect.signature(func)
    params = {}
    
    for param_name, param in sig.parameters.items():
        param_info = {"type": "string"}  # Default type
        
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == float:
                param_info["type"] = "number"
            elif param.annotation == int:
                param_info["type"] = "integer"
            elif param.annotation == bool:
                param_info["type"] = "boolean"
        
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default
        
        params[param_name] = param_info
    
    return {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": params
    }


# ============================================================================
# HTTP ENDPOINTS
# ============================================================================

@app.get("/")
async def health_check():
    """Health check endpoint."""
    tool_count = len(mcp._tools) if hasattr(mcp, '_tools') else 0
    return {
        "status": "healthy",
        "server": "mcp-http-server",
        "version": "1.0.0",
        "transport": "HTTP",
        "tools_count": tool_count
    }


@app.get("/mcp/tools/list")
async def list_tools():
    """
    List all available MCP tools.
    
    Returns:
        List of tool definitions with names, descriptions, and parameters
    """
    print("[API] GET /mcp/tools/list")
    
    tools = []
    if hasattr(mcp, '_tools'):
        for tool_func in mcp._tools:
            tools.append(get_tool_info(tool_func))
    
    return {"tools": tools}


@app.post("/mcp/tools/call")
async def call_tool(request: ToolCallRequest):
    """
    Call a specific MCP tool.
    
    Args:
        request: Tool call request with name and arguments
        
    Returns:
        Tool execution result
    """
    print(f"[API] POST /mcp/tools/call: {request.name}")
    
    try:
        # Get the tool function
        tool_func = get_tool_function(request.name)
        
        if tool_func is None:
            raise HTTPException(
                status_code=404,
                detail=f"Tool not found: {request.name}"
            )
        
        # Execute the tool
        result = tool_func(**request.arguments)
        
        print(f"[API] Tool result: {result}")
        
        return {"result": result}
        
    except TypeError as e:
        # Parameter errors
        print(f"[API] Parameter error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid parameters: {str(e)}",
                "tool": request.name
            }
        )
    except Exception as e:
        # Unexpected errors
        print(f"[API] Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Internal server error: {str(e)}",
                "tool": request.name
            }
        )


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on server startup."""
    print()
    print("=" * 70)
    print("MCP HTTP SERVER - STARTING")
    print("=" * 70)
    
    tool_count = len(mcp._tools) if hasattr(mcp, '_tools') else 0
    print(f"✓ Server name: {mcp.name}")
    print(f"✓ Tools registered: {tool_count}")
    print()
    
    if hasattr(mcp, '_tools'):
        for tool_func in mcp._tools:
            print(f"  • {tool_func.__name__}: {tool_func.__doc__ or 'No description'}")
    print()
    
    print("=" * 70)
    print("SERVER READY")
    print("=" * 70)
    print()
    print("📍 Server running at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("📖 ReDoc available at: http://localhost:8000/redoc")
    print()
    print("🔧 Test endpoints:")
    print("   curl http://localhost:8000/")
    print("   curl http://localhost:8000/mcp/tools/list")
    print()
    print("💡 Or run: ./test_api.sh")
    print()
    print("=" * 70)
    print()


@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown."""
    print()
    print("=" * 70)
    print("MCP HTTP SERVER - SHUTTING DOWN")
    print("=" * 70)
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the HTTP server."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WELCOME TO MCP DEMO 06" + " " * 31 + "║")
    print("║" + " " * 20 + "HTTP Transport" + " " * 34 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📚 This demo demonstrates:")
    print("   • MCP server using HTTP transport")
    print("   • RESTful API endpoints for MCP")
    print("   • CORS support for web clients")
    print("   • FastAPI with automatic OpenAPI docs")
    print("   • FastMCP + FastAPI integration")
    print()
    
    print("🚀 Starting server...")
    print()
    
    # Run FastAPI server with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()

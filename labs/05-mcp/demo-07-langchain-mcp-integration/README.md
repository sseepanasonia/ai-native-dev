# Demo 07: LangChain MCP Integration 🔗

Learn how to integrate MCP tools with LangChain agents to build powerful AI workflows.

## 🎯 What You'll Learn

- Converting MCP tools to LangChain tools
- Building LangChain agents with MCP tools
- Framework interoperability patterns
- Tool calling with OpenAI function calling
- Multi-tool orchestration
- Agent-based task execution

## 📦 What's Inside

✅ **MCPTool Wrapper** - Bridge MCP to LangChain  
✅ **LangChain Agent** - Use MCP tools in agents  
✅ **Calculator + Weather** - Multiple MCP servers  
✅ **OpenAI Integration** - GPT-4 function calling  
✅ **Async Support** - Non-blocking operations  
✅ **Real Examples** - Practical agent workflows

## 🚀 Quick Start

### 1. Get OpenAI API Key

Sign up at https://platform.openai.com/

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API key
```

### 4. Run the Demo

```bash
uv run python main.py
```

## 🧠 Key Concepts

### Why Integrate MCP with LangChain?

**Benefits:**

- **Reusability**: Write MCP tools once, use everywhere
- **Standardization**: Common protocol across frameworks
- **Composability**: Mix MCP and native LangChain tools
- **Ecosystem**: Access MCP community tools

### Architecture

```
┌─────────────────────────────┐
│   LangChain Agent           │
│   (Orchestration)           │
└──────────┬──────────────────┘
           │
           │ MCPTool Wrapper
           │
┌──────────▼──────────────────┐
│   MCP Server(s)             │
│   (Tool Providers)          │
└─────────────────────────────┘
```

### MCPTool Wrapper

Converts MCP tools to LangChain `BaseTool`:

```python
class MCPTool(BaseTool):
    """LangChain tool wrapper for MCP tools."""

    name: str
    description: str
    mcp_client: Client
    mcp_tool_name: str

    def _run(self, **kwargs) -> str:
        # Execute MCP tool
        result = await self.mcp_client.call_tool(
            self.mcp_tool_name,
            kwargs
        )
        return str(result)
```

## 📚 Example Workflows

### 1. Calculator Agent

```python
agent = create_agent_with_mcp_tools()

result = agent.run(
    "Calculate (15 + 27) * 3 and then divide by 6"
)
# Agent uses: add → multiply → divide
```

### 2. Weather Analysis

```python
result = agent.run(
    "What's the temperature difference between London and Paris?"
)
# Agent uses: get_weather(London) → get_weather(Paris) → subtract
```

### 3. Multi-Step Tasks

```python
result = agent.run(
    "Find cities where temperature is above 20°C: London, Paris, Berlin"
)
# Agent uses: get_weather multiple times → compare results
```

## 🔧 Advanced Usage

### Multiple MCP Servers

```python
toolkit = MCPToolkit(mcp_servers=[
    {"command": "python", "args": ["calculator_server.py"]},
    {"command": "python", "args": ["weather_server.py"]},
    {"command": "python", "args": ["database_server.py"]}
])

tools = await toolkit.get_tools()
# All tools from all servers available to agent
```

### Custom Tool Selection

```python
# Use only specific tools
selected_tools = [
    tool for tool in tools
    if tool.name in ["add", "subtract", "get_weather"]
]

agent = create_agent(llm, selected_tools)
```

## 🎓 Learning Notes

### LangChain Agent Types

1. **Zero-shot ReAct**: Best for MCP tools (reasons and acts)
2. **OpenAI Functions**: Native function calling
3. **Structured Chat**: For complex multi-tool scenarios

### Tool Calling Flow

```
1. User Query → Agent
2. Agent → LLM (What tools needed?)
3. LLM → Agent (Use tool X with args Y)
4. Agent → MCPTool → MCP Server
5. MCP Server → Tool Result
6. Result → Agent → LLM
7. LLM → Final Answer
```

### Error Handling

```python
try:
    result = agent.run(query)
except ValueError as e:
    # Tool execution error
    print(f"Tool error: {e}")
except Exception as e:
    # Agent/LLM error
    print(f"Agent error: {e}")
```

## 📁 Project Structure

```
demo-07-langchain-mcp-integration/
├── .python-version      # Python 3.12
├── .env                 # API keys
├── .env.example         # Template
├── .gitignore          # Excludes .env
├── pyproject.toml      # Dependencies
├── README.md           # This file
└── main.py             # Integration demo
```

## 🔧 Troubleshooting

### OpenAI API Error

Make sure your API key is valid and has credits.

### Tool Not Available

Check MCP server is running and tools are registered.

### Import Errors

```bash
# Make sure all dependencies installed
uv sync
```

## 💡 Exercise Ideas

1. **RAG + MCP**: Combine LangChain RAG with MCP tools
2. **Custom Tools**: Create your own MCP tools for LangChain
3. **Multi-Agent**: Build agent that coordinates other agents
4. **Streaming**: Add streaming responses for real-time feedback

## 📚 Next Steps

1. **Demo 08**: Build autonomous MCP agent
2. **Demo 09**: Multi-server coordination
3. Check LangChain docs for more agent patterns

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [MCP Specification](https://spec.modelcontextprotocol.io/)

---

**Happy Learning! 🚀**

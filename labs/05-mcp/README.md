# MCP (Model Context Protocol) Demos 🚀

A comprehensive series of demonstrations for learning the Model Context Protocol (MCP) using **FastMCP** - from basics to advanced agent integration.

## 📚 Overview

These demos progressively teach you how to build, deploy, and integrate MCP servers for AI applications using **FastMCP**, a Pythonic framework that makes MCP development simple and intuitive. Each demo is self-contained with full documentation, working code, and learning exercises.

**Why FastMCP?**

- ✅ **Pythonic decorators** - Familiar `@mcp.tool()` syntax
- ✅ **Automatic validation** - Type hints generate JSON schemas
- ✅ **60% less code** - Compared to raw MCP protocol
- ✅ **Beginner friendly** - Less boilerplate, clearer patterns
- ✅ **Production ready** - Built on the official MCP specification

---

## 🎯 Demo Progression

### **Fundamentals** (Demos 01-03)

#### [Demo 01: Basic MCP Stdio](demo-01-basic-mcp-stdio/)

**Level**: Beginner  
**Topics**: FastMCP basics, stdio transport, tool definition  
**Time**: 15 minutes

Learn the core concepts of MCP with FastMCP:

- Server initialization with `FastMCP()`
- Tool registration with `@mcp.tool()` decorator
- Client-server communication via stdio
- JSON-RPC message exchange
- Tool discovery and invocation

**What you'll build**: Simple echo and greet tools

**Key Code**:

```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet someone"""
    return f"Hello, {name}!"

mcp.run()
```

---

#### [Demo 02: Calculator Tools](demo-02-mcp-calculator-tools/)

**Level**: Beginner  
**Topics**: Multiple tools, parameter validation, error handling  
**Time**: 20 minutes

Build a calculator server with multiple tools:

- 8 calculator operations (add, subtract, multiply, divide, power, modulo, calculate, info)
- Type validation with Python type hints
- Error handling for edge cases (division by zero)
- Using `Literal` types for enumerated parameters
- Tool organization best practices

**What you'll build**: Full-featured calculator MCP server

**Key Learnings**:

- Multiple tool registration
- Structured return values
- Error handling patterns
- Literal types for enums

---

#### [Demo 03: Weather API Server](demo-03-mcp-weather-server/)

**Level**: Intermediate  
**Topics**: External API integration, async I/O, environment config  
**Time**: 30 minutes

Integrate real-world APIs via MCP:

- OpenWeatherMap API integration
- Async HTTP requests with httpx
- Environment variable configuration
- API key security
- Demo mode for testing without API key
- Error handling for network failures

**What you'll build**: Weather lookup, geocoding, and comparison tools

**Requires**: OpenWeatherMap API key (free) - optional, has demo mode

**Key Learnings**:

- External API calls in MCP tools
- Async/await patterns
- Environment-based configuration
- Graceful error handling

---

### **Custom Servers** (Demos 04-05)

#### [Demo 04: Filesystem Server](demo-04-mcp-filesystem-server/)

**Level**: Intermediate  
**Topics**: File operations, security, path validation  
**Time**: 30 minutes

Build secure filesystem tools:

- Read/write files within sandbox
- Directory listing and file search
- Path traversal prevention
- Async file I/O with aiofiles
- Security validation helpers

**What you'll build**: Secure file management server

**Key Learnings**:

- Security-first design
- Path validation patterns
- Async file operations
- Sandboxing for safety

---

#### Demo 05: Database Server _(Coming Soon)_

**Level**: Intermediate  
**Topics**: Database integration, SQL security, connection pooling

Connect databases via MCP:

- SQLite/PostgreSQL integration
- Safe parameterized queries
- Table schema introspection
- Database resources

---

### **Alternative Transports** (Demo 06)

#### [Demo 06: HTTP Transport](demo-06-mcp-http-transport/)

**Level**: Intermediate  
**Topics**: HTTP/REST API, FastAPI, FastMCP integration  
**Time**: 30 minutes

Deploy MCP over HTTP instead of stdio:

- FastMCP + FastAPI integration
- RESTful endpoints for tool calling
- CORS support for web clients
- Health checks and monitoring
- OpenAPI documentation
- Web deployment patterns

**What you'll build**: HTTP-based MCP server with REST API

**Key Learnings**:

- HTTP vs stdio transport
- FastMCP/FastAPI interoperability
- Web API best practices
- CORS configuration

---

### **Framework Integration** (Demo 07)

#### [Demo 07: LangChain Integration](demo-07-langchain-mcp-integration/)

**Level**: Advanced  
**Topics**: Framework interoperability, LangChain agents, tool orchestration  
**Time**: 45 minutes

Connect FastMCP tools with LangChain agents:

- Convert FastMCP tools to LangChain tools
- Build OpenAI function-calling agents
- Multi-tool orchestration
- Agent-based workflows
- Cross-framework patterns

**What you'll build**: LangChain agent with FastMCP calculator tools

**Requires**: OpenAI API key

**Key Learnings**:

- Framework adapter patterns
- Tool wrapping techniques
- Agent orchestration
- Multi-step reasoning

---

### **Advanced Patterns** (Demos 08-10)

#### Demo 08: MCP Agent _(Coming Soon)_

**Level**: Advanced  
**Topics**: Autonomous agents, planning, multi-step execution

Build an autonomous agent:

- Agent loop implementation
- Planning and execution
- Tool selection strategies
- State management
- Error recovery

---

#### Demo 09: Multi-MCP Servers _(Coming Soon)_

**Level**: Advanced  
**Topics**: Server coordination, tool namespacing, resource sharing

Coordinate multiple MCP servers:

- Load multiple servers
- Namespace tools to avoid conflicts
- Share resources between servers
- Routing and orchestration

---

#### Demo 10: Resources & Prompts _(Coming Soon)_

**Level**: Advanced  
**Topics**: MCP resources, prompt templates, context management

Beyond tools - resources and prompts:

- Expose files as resources
- Dynamic resource generation
- Prompt templates
- Context integration

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **uv** package manager
- Basic Python knowledge
- Understanding of async/await (helpful)

### Quick Start

1. **Navigate to a demo**:

   ```bash
   cd demo-01-basic-mcp-stdio
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Run the demo**:

   ```bash
   uv run python main.py
   ```

4. **Test with MCP Inspector**:
   ```bash
   npx @modelcontextprotocol/inspector uv run python main.py
   ```

---

## 📖 Learning Path

### 🎓 **Beginner Path** (2-3 hours)

Start here if you're new to MCP:

1. [Demo 01](demo-01-basic-mcp-stdio/) - Learn FastMCP basics
2. [Demo 02](demo-02-mcp-calculator-tools/) - Multiple tools and validation
3. [Demo 03](demo-03-mcp-weather-server/) - External API integration
4. Practice exercises in each demo

By the end, you'll understand:

- FastMCP core concepts
- Tool creation patterns
- Type hints and validation
- External API integration

### 🔥 **Intermediate Path** (4-5 hours)

Continue with:

5. [Demo 04](demo-04-mcp-filesystem-server/) - Secure file operations
6. [Demo 06](demo-06-mcp-http-transport/) - HTTP deployment
7. Build your own MCP server

You'll master:

- Security best practices
- Alternative transports
- Production patterns
- Custom tool design

### 🚀 **Advanced Path** (6+ hours)

Dive into:

8. [Demo 07](demo-07-langchain-mcp-integration/) - LangChain agents
9. Demo 08 - Autonomous agents (coming soon)
10. Demo 09 - Multi-server coordination (coming soon)
11. Demo 10 - Resources and prompts (coming soon)

You'll learn:

- Framework integration
- Agent orchestration
- Complex workflows
- Production deployment

---

## 🎯 Key Concepts

### What is MCP?

**Model Context Protocol (MCP)** is an open protocol that standardizes how AI applications connect to external tools and data sources. Think of it as "USB for AI" - a universal way to plug tools into LLMs.

### What is FastMCP?

**FastMCP** is a Pythonic framework for building MCP servers that:

- Uses familiar Python decorators (`@mcp.tool()`)
- Leverages type hints for automatic validation
- Reduces boilerplate by 60%
- Makes MCP development accessible to Python developers

### Why MCP?

**Before MCP**:

- Every AI framework has its own tool API
- Tools must be rewritten for each framework
- Integration is complex and fragmented

**With MCP**:

- ✅ Write tools once, use everywhere
- ✅ Standard protocol for all frameworks
- ✅ Easy integration with Claude, LangChain, custom apps
- ✅ Separation of concerns (tools vs. application logic)

### FastMCP vs Raw MCP

| Aspect             | Raw MCP                     | FastMCP                      |
| ------------------ | --------------------------- | ---------------------------- |
| **Code Length**    | ~100 lines                  | ~40 lines (60% less)         |
| **Learning Curve** | Steep (protocol details)    | Gentle (Pythonic decorators) |
| **Type Safety**    | Manual schema generation    | Automatic from type hints    |
| **Boilerplate**    | High (stdio setup, schemas) | Low (handled for you)        |
| **Best For**       | Protocol implementers       | Application developers       |

### Core Components

```python
from fastmcp import FastMCP

# 1. Create server instance
mcp = FastMCP("Server Name")

# 2. Define tools with decorators
@mcp.tool()
def my_tool(param: str) -> dict:
    """Tool description"""
    return {"result": "value"}

# 3. Run server
mcp.run()
```

**That's it!** FastMCP handles:

- JSON-RPC protocol
- Stdio communication
- Schema generation
- Type validation
- Error handling

---

## 💡 Common Patterns

### Pattern 1: Simple Tool

```python
@mcp.tool()
def greet(name: str) -> str:
    """Greet someone"""
    return f"Hello, {name}!"
```

### Pattern 2: Structured Response

```python
@mcp.tool()
def calculate(a: float, b: float) -> dict:
    """Add two numbers"""
    return {
        "result": a + b,
        "expression": f"{a} + {b} = {a + b}"
    }
```

### Pattern 3: Error Handling

```python
@mcp.tool()
def divide(a: float, b: float) -> dict:
    """Divide two numbers"""
    if b == 0:
        return {"error": "Division by zero"}
    return {"result": a / b}
```

### Pattern 4: Async Operations

```python
@mcp.tool()
async def fetch_data(url: str) -> dict:
    """Fetch data from URL"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### Pattern 5: Enum Parameters

```python
from typing import Literal

@mcp.tool()
def calculate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float
) -> dict:
    """Perform calculation"""
    # Implementation
```

---

## 🛠️ Tools & Resources

### Essential Tools

- **[MCP Inspector](https://github.com/modelcontextprotocol/inspector)** - Test MCP servers interactively
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package manager
- **[FastMCP](https://github.com/jlowin/fastmcp)** - Pythonic MCP framework

### Documentation

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Official Docs](https://modelcontextprotocol.io/)

### Community

- [MCP Discord](https://discord.gg/modelcontextprotocol)
- [GitHub Discussions](https://github.com/modelcontextprotocol/specification/discussions)

---

## 🎓 Best Practices

### Tool Design

1. **Single Responsibility**: Each tool does one thing well
2. **Clear Descriptions**: Write helpful docstrings
3. **Type Hints**: Use proper Python types for validation
4. **Error Handling**: Return errors gracefully, don't raise exceptions
5. **Structured Output**: Return dicts with consistent keys

### Security

1. **Validate Inputs**: Use type hints and custom validation
2. **Sandbox Operations**: Restrict file/network access
3. **API Keys**: Store in environment variables, never hardcode
4. **Path Security**: Validate and sanitize file paths
5. **Rate Limiting**: Protect against abuse

### Performance

1. **Use Async**: For I/O-bound operations
2. **Connection Pooling**: Reuse database/HTTP connections
3. **Caching**: Cache expensive operations
4. **Timeouts**: Set reasonable timeouts for external calls
5. **Resource Cleanup**: Use context managers

---

## 🤝 Contributing

Found an issue or want to add a demo? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Follow the existing demo structure
4. Add comprehensive README with exercises
5. Submit a pull request

---

## 📄 License

MIT License - feel free to use these demos for learning and teaching!

---

**Happy Learning! 🚀**

For questions or feedback, open an issue or reach out on Discord.

**What you'll build**: HTTP-based MCP server

**When to use**:

- ✅ Web applications, mobile apps
- ✅ Remote/cloud deployments
- ✅ Microservices architecture
- ❌ Desktop apps (use stdio)

---

### **Framework Integration** (Demo 07)

#### [Demo 07: LangChain Integration](demo-07-langchain-mcp-integration/)

**Level**: Advanced  
**Topics**: Framework interoperability, agents, tool orchestration  
**Time**: 45 minutes

Use MCP tools in LangChain agents:

- Converting MCP tools to LangChain format
- Building agents with MCP tool access
- Multi-tool orchestration
- OpenAI function calling
- Agent-based workflows

**What you'll build**: LangChain agent using MCP calculator tools

**Requires**: OpenAI API key

---

### **Advanced Patterns** (Demos 08-10)

#### Demo 08: MCP Agent _(Coming Soon)_

**Level**: Advanced  
**Topics**: Autonomous agents, ReAct pattern, multi-step reasoning

Build an autonomous agent:

- ReAct loop (Reason → Act → Observe)
- Multi-step task execution
- Tool selection logic
- State management

---

#### Demo 09: Multi-MCP Servers _(Coming Soon)_

**Level**: Advanced  
**Topics**: Server composition, microservices, coordination

Connect multiple specialized servers:

- Database + Email + Analytics servers
- Coordinated operations
- Server lifecycle management
- Microservices patterns

---

#### Demo 10: Resources & Prompts _(Coming Soon)_

**Level**: Advanced  
**Topics**: Complete MCP capabilities, resources, prompts

Explore full MCP protocol:

- Resources (data as URIs)
- Prompt templates
- Resource subscriptions
- Complete feature demonstration

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+** (Check: `python --version`)
- **uv package manager** (Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Basic Python knowledge**
- **API keys** (for some demos):
  - OpenWeatherMap (Demo 03) - [Get free key](https://openweathermap.org/api)
  - OpenAI (Demo 07) - [Get key](https://platform.openai.com/)

### Quick Start

```bash
# Navigate to any demo
cd demo-01-basic-mcp-stdio

# Install dependencies
uv sync

# Run the demo
uv run python main.py
```

Each demo is self-contained and can be run independently.

---

## 📖 Learning Path

### Path 1: Quick Introduction (1 hour)

Perfect if you want a rapid overview:

1. Demo 01 - Basics
2. Demo 02 - Multiple tools
3. Demo 06 - HTTP transport

### Path 2: Comprehensive Learning (4 hours)

Complete understanding of MCP:

1. Demo 01 - Basics
2. Demo 02 - Calculator
3. Demo 03 - Weather API
4. Demo 04 - Filesystem
5. Demo 06 - HTTP
6. Demo 07 - LangChain

### Path 3: Production Readiness (6 hours)

Everything needed for real applications:

- All demos 01-10
- Complete exercises
- Build custom tools

---

## 🎓 Key Concepts Covered

### Core MCP

- ✅ Server and client architecture
- ✅ Tool definition and registration
- ✅ JSON-RPC 2.0 protocol
- ✅ Type safety with JSON Schema
- ✅ Error handling patterns

### Transports

- ✅ Stdio (local, desktop apps)
- ✅ HTTP (web, remote, cloud)
- ⏳ SSE (Server-Sent Events)
- ⏳ WebSockets

### Integration

- ✅ External APIs (REST, GraphQL)
- ✅ Filesystems (with security)
- ⏳ Databases (SQL, NoSQL)
- ✅ LangChain agents
- ⏳ Multi-server coordination

### Advanced Topics

- ✅ Security and validation
- ✅ Async operations
- ✅ Framework interoperability
- ⏳ Resources and prompts
- ⏳ Autonomous agents

---

## 🔧 Common Issues & Solutions

### Import Error: No module named 'mcp'

```bash
# Make sure you're in the demo directory
cd demo-XX-name

# Install dependencies
uv sync
```

### API Key Not Found

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your key
nano .env
```

### Port Already in Use (Demo 06)

```bash
# Kill process on port 8000
lsof -i :8000
kill <PID>

# Or use different port
uvicorn main:app --port 8001
```

---

## 📚 Additional Resources

### Official Documentation

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP GitHub](https://github.com/modelcontextprotocol)
- [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)

### Related Guides

- [Guide 07: MCP Comprehensive Guide](../../guides/07_mcp_guide.md)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Community

- [MCP Discord](https://discord.gg/mcp)
- [GitHub Discussions](https://github.com/modelcontextprotocol/discussions)

---

## 💡 Exercise Ideas

After completing the demos, try building:

1. **GitHub MCP Server** - Repository tools (create issue, search code)
2. **Slack MCP Server** - Send messages, search conversations
3. **Email MCP Server** - Send emails, check inbox
4. **Calendar MCP Server** - Create events, check availability
5. **Analytics MCP Server** - Query metrics, generate reports

---

## 🎯 Next Steps

After mastering these demos:

1. **Build Your Own**: Create MCP servers for your specific use case
2. **Integrate with Agents**: Use MCP in production AI applications
3. **Contribute**: Share your MCP servers with the community
4. **Explore Practices**: Check `practices/` folder for projects

---

## 🤝 Need Help?

- **Demo Issues**: Check individual demo READMEs
- **General Questions**: Review [MCP Guide](../../guides/07_mcp_guide.md)
- **Bugs**: Check code comments and error messages
- **Advanced Topics**: Explore official MCP documentation

---

## 📊 Demo Comparison

| Demo            | Level        | Time  | API Key        | Transport | Output   |
| --------------- | ------------ | ----- | -------------- | --------- | -------- |
| 01 - Basic      | Beginner     | 15min | ❌             | Stdio     | Console  |
| 02 - Calculator | Beginner     | 20min | ❌             | Stdio     | Console  |
| 03 - Weather    | Intermediate | 30min | ✅ OpenWeather | Stdio     | Console  |
| 04 - Filesystem | Intermediate | 30min | ❌             | Stdio     | Console  |
| 05 - Database   | Intermediate | 30min | ❌             | Stdio     | Console  |
| 06 - HTTP       | Intermediate | 30min | ❌             | HTTP      | JSON API |
| 07 - LangChain  | Advanced     | 45min | ✅ OpenAI      | Stdio     | Agent    |
| 08 - Agent      | Advanced     | 45min | ✅             | Stdio     | Agent    |
| 09 - Multi      | Advanced     | 60min | ❌             | Stdio     | Multi    |
| 10 - Resources  | Advanced     | 45min | ❌             | Stdio     | Console  |

---

**Happy Learning! 🚀**

_Part of the AI-Native Development Course_

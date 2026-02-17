# Demo 06: MCP HTTP Transport 🌐

Learn how to build MCP servers using HTTP/REST transport for web-based and remote access.

## 🎯 What You'll Learn

- HTTP transport vs stdio transport
- Building RESTful MCP APIs with FastAPI
- CORS configuration for web clients
- HTTP error handling
- When to use HTTP vs stdio
- API documentation with OpenAPI

## 📦 What's Inside

✅ **HTTP Endpoints** - `/mcp/tools/list`, `/mcp/tools/call`  
✅ **FastAPI Server** - Production-ready web server  
✅ **CORS Support** - Cross-origin requests for browsers  
✅ **Health Checks** - Monitoring and status endpoints  
✅ **Error Handling** - HTTP status codes and error messages  
✅ **OpenAPI Docs** - Auto-generated API documentation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Start the Server

```bash
uv run python main.py
```

Server starts on http://localhost:8000

### 3. Test the API

**Option A: Use test script**

```bash
chmod +x test_api.sh
./test_api.sh
```

**Option B: Manual curl commands**

```bash
# Health check
curl http://localhost:8000/

# List tools
curl http://localhost:8000/mcp/tools/list

# Call a tool
curl -X POST http://localhost:8000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "greet", "arguments": {"name": "World"}}'
```

### 4. View API Docs

Open in browser:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 API Endpoints

### GET /

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "server": "mcp-http-server",
  "version": "1.0.0"
}
```

### GET /mcp/tools/list

List all available tools.

**Response:**

```json
{
  "tools": [
    {
      "name": "greet",
      "description": "Greet someone",
      "inputSchema": {...}
    }
  ]
}
```

### POST /mcp/tools/call

Call a specific tool.

**Request:**

```json
{
  "name": "add",
  "arguments": {
    "a": 5,
    "b": 3
  }
}
```

**Response:**

```json
{
  "result": 8.0
}
```

**Error Response:**

```json
{
  "error": "Cannot divide by zero",
  "tool": "divide",
  "status": "error"
}
```

## 🧠 Key Concepts

### HTTP vs Stdio Transport

| Feature        | Stdio             | HTTP                    |
| -------------- | ----------------- | ----------------------- |
| **Deployment** | Same machine only | Remote/cloud            |
| **Latency**    | Ultra-low (~μs)   | Higher (~ms)            |
| **Use Case**   | Desktop apps, CLI | Web apps, microservices |
| **Setup**      | Simple            | Requires web server     |
| **Scaling**    | Single instance   | Load balancing          |
| **Security**   | Process isolation | TLS, authentication     |
| **Web Access** | ❌ No             | ✅ Yes                  |

### When to Use HTTP

Choose HTTP transport when you need:

- **Remote Access**: Client and server on different machines
- **Web Integration**: Browser-based clients
- **Scaling**: Multiple server instances behind load balancer
- **Standard APIs**: RESTful interface familiar to developers
- **Monitoring**: HTTP health checks and metrics

Choose Stdio transport when you need:

- **Desktop Apps**: VS Code extensions, Claude Desktop
- **CLI Tools**: Command-line applications
- **Low Latency**: Microsecond response times
- **Simple Deployment**: No web server configuration

## 🔐 Security Considerations

### CORS (Cross-Origin Resource Sharing)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Production: specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Authentication

Add API keys for production:

```python
from fastapi import Header, HTTPException

async def verify_token(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/mcp/tools/call", dependencies=[Depends(verify_token)])
async def call_tool(...):
    ...
```

### HTTPS/TLS

For production, always use HTTPS:

```bash
uvicorn main:app --host 0.0.0.0 --port 443 \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem
```

## 📁 Project Structure

```
demo-06-mcp-http-transport/
├── .python-version      # Python 3.12
├── .gitignore          # Standard ignores
├── pyproject.toml      # Dependencies
├── README.md           # This file
├── main.py             # HTTP server implementation
└── test_api.sh         # API test script
```

## 🔧 Troubleshooting

### Port Already in Use

```
ERROR: [Errno 48] Address already in use
```

**Solution:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill <PID>

# Or use different port
uvicorn main:app --port 8001
```

### CORS Errors in Browser

```
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:3000'
has been blocked by CORS policy
```

**Solution:** CORS is already configured in the demo. For production, specify allowed origins:

```python
allow_origins=["https://yourdomain.com"]
```

### Connection Refused

**Solution:** Make sure server is running:

```bash
curl http://localhost:8000/
```

## 🎓 Learning Notes

### FastAPI Benefits

1. **Automatic Validation**: Pydantic models validate requests
2. **OpenAPI Docs**: Interactive API documentation
3. **Async Support**: Native async/await
4. **Type Safety**: Python type hints
5. **Performance**: Built on Starlette (fast!)

### HTTP Status Codes

- `200 OK`: Successful tool call
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Tool doesn't exist
- `500 Internal Server Error`: Tool execution failed

### Streaming Responses

For long-running operations:

```python
from fastapi.responses import StreamingResponse

@app.post("/mcp/tools/call/stream")
async def call_tool_streaming(request: ToolCallRequest):
    async def generate():
        async for chunk in execute_tool_streaming(request.name, request.arguments):
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

## 💡 Exercise Ideas

1. **Add Authentication**: Implement API key validation
2. **Rate Limiting**: Prevent abuse with slowapi
3. **Metrics**: Add Prometheus metrics endpoint
4. **Caching**: Cache tool responses with Redis
5. **WebSockets**: Real-time bidirectional communication

## 📚 Next Steps

1. **Demo 07**: Integrate with LangChain agents
2. **Demo 08**: Build autonomous AI agent
3. **Demo 09**: Combine multiple MCP servers

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Uvicorn Docs](https://www.uvicorn.org/)

## 🤝 Need Help?

- Check server is running: `curl http://localhost:8000/`
- View API docs: http://localhost:8000/docs
- Check logs for errors
- Use `./test_api.sh` to test all endpoints

---

**Happy Learning! 🚀**

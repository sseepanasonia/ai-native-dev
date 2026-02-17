# Demo 04: MCP Filesystem Server 📁

Learn how to build a secure MCP server for filesystem operations with proper path validation and security measures.

## 🎯 What You'll Learn

- Building domain-specific MCP servers
- Path security and validation (prevent directory traversal)
- Async file operations with aiofiles
- Exposing files as MCP resources
- MIME type detection
- Safe sandbox operations

## 📦 What's Inside

✅ **read_file** - Read file contents securely  
✅ **write_file** - Create/update files with validation  
✅ **list_directory** - List directory contents  
✅ **search_files** - Find files by pattern (glob)  
✅ **delete_file** - Remove files safely  
✅ **Path Security** - Prevent directory traversal attacks

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run demo
uv run python main.py
```

## 📚 Available Tools

### read_file(path: str) -> str

Read and return file contents.

**Security**: Only files within sandbox/ directory

### write_file(path: str, content: str) -> dict

Write content to file (creates parent directories).

### list_directory(path: str = ".") -> list

List files and directories.

### search_files(pattern: str, path: str = ".") -> list

Search for files matching glob pattern.

## 🔐 Security Features

```python
# Path validation prevents attacks:
# ✅ Allowed: "sandbox/notes.txt"
# ❌ Blocked: "../../../etc/passwd"
# ❌ Blocked: "/absolute/path"
```

## 🎓 Key Concepts

- **Sandboxing**: Restrict operations to safe directory
- **Path Validation**: Check all paths are within sandbox
- **Async I/O**: Non-blocking file operations
- **Resources**: Expose files as discoverable resources

## 📚 Next Steps

- Demo 05: Database server
- Demo 06: HTTP transport
- Demo 09: Combine with other servers

---

**Happy Learning! 🚀**

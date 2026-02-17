"""
Demo 04: MCP Filesystem Server

Secure filesystem operations via MCP:
- Read/write files within sandbox
- List and search directories  
- Path security validation
- Async file I/O

Key Concepts:
- Sandboxing for security
- Path traversal prevention
- Async file operations with aiofiles
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, List
from fastmcp import FastMCP
import aiofiles

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sandbox directory - all operations restricted to this path
SANDBOX_DIR = Path(__file__).parent / "sandbox"
SANDBOX_DIR.mkdir(exist_ok=True)

mcp = FastMCP("Filesystem Server")

print("=" * 70)
print("MCP DEMO 04: FILESYSTEM SERVER")
print("=" * 70)
print(f"✓ Sandbox directory: {SANDBOX_DIR}")
print()

# ============================================================================
# SECURITY HELPERS
# ============================================================================

def validate_path(path: str) -> Path:
    """
    Validate that path is within sandbox directory.
    
    Prevents directory traversal attacks like:
    - ../../../etc/passwd
    - /absolute/paths
    
    Args:
        path: Relative path from sandbox
        
    Returns:
        Resolved absolute path within sandbox
        
    Raises:
        ValueError: If path escapes sandbox
    """
    full_path = (SANDBOX_DIR / path).resolve()
    
    # Check if resolved path is within sandbox
    try:
        full_path.relative_to(SANDBOX_DIR.resolve())
    except ValueError:
        raise ValueError(f"Access denied: path outside sandbox directory")
    
    return full_path

# ============================================================================
# FILESYSTEM TOOLS
# ============================================================================

@mcp.tool()
async def read_file(path: str) -> dict:
    """
    Read and return file contents.
    
    Args:
        path: Path relative to sandbox directory
        
    Returns:
        Dictionary with file content and metadata
    """
    print(f"[Server] read_file: {path}")
    
    try:
        full_path = validate_path(path)
        
        if not full_path.exists():
            return {"error": f"File not found: {path}"}
        
        if not full_path.is_file():
            return {"error": f"Not a file: {path}"}
        
        async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        print(f"[Server] Read {len(content)} bytes from {path}")
        return {
            "content": content,
            "path": path,
            "size": len(content)
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def write_file(path: str, content: str) -> dict:
    """
    Write content to file (creates parent directories if needed).
    
    Args:
        path: Path relative to sandbox directory
        content: Content to write
        
    Returns:
        Dictionary with success status and bytes written
    """
    print(f"[Server] write_file: {path} ({len(content)} bytes)")
    
    try:
        full_path = validate_path(path)
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        print(f"[Server] Wrote {len(content)} bytes to {path}")
        
        return {
            "success": True,
            "path": path,
            "bytes_written": len(content)
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def list_directory(path: str = ".") -> dict:
    """
    List contents of directory.
    
    Args:
        path: Path relative to sandbox directory (default: ".")
        
    Returns:
        Dictionary with items list and count
    """
    print(f"[Server] list_directory: {path}")
    
    try:
        full_path = validate_path(path)
        
        if not full_path.exists():
            return {"error": f"Directory not found: {path}"}
        
        if not full_path.is_dir():
            return {"error": f"Not a directory: {path}"}
        
        items = []
        for item in full_path.iterdir():
            rel_path = item.relative_to(SANDBOX_DIR)
            items.append({
                "name": item.name,
                "path": str(rel_path),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })
        
        print(f"[Server] Found {len(items)} items in {path}")
        return {
            "items": items,
            "count": len(items),
            "path": path
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def search_files(pattern: str, path: str = ".") -> dict:
    """
    Search for files matching glob pattern.
    
    Args:
        pattern: Glob pattern (e.g., "*.txt", "**/*.py")
        path: Starting directory relative to sandbox
        
    Returns:
        Dictionary with matching file paths and count
    """
    print(f"[Server] search_files: pattern={pattern}, path={path}")
    
    try:
        full_path = validate_path(path)
        
        if not full_path.exists():
            return {"error": f"Directory not found: {path}"}
        
        matches = list(full_path.rglob(pattern))
        
        # Convert to relative paths
        rel_matches = [str(m.relative_to(SANDBOX_DIR)) for m in matches if m.is_file()]
        
        print(f"[Server] Found {len(rel_matches)} matches for {pattern}")
        return {
            "matches": rel_matches,
            "count": len(rel_matches),
            "pattern": pattern
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def delete_file(path: str) -> dict:
    """
    Delete a file.
    
    Args:
        path: Path relative to sandbox directory
        
    Returns:
        Dictionary with success status
    """
    print(f"[Server] delete_file: {path}")
    
    try:
        full_path = validate_path(path)
        
        if not full_path.exists():
            return {"error": f"File not found: {path}"}
        
        if not full_path.is_file():
            return {"error": f"Not a file: {path}"}
        
        full_path.unlink()
        
        print(f"[Server] Deleted {path}")
        return {"success": True, "path": path}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    mcp.run()

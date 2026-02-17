# Demo 02: MCP Calculator Tools 🧮

## What You'll Learn 📚

In this demo, you'll learn:

- How to create **multiple tools** in a single MCP server
- **Input validation** and **error handling** patterns
- Using Python **type hints** for automatic validation
- Best practices for tool **documentation**
- The **Literal type** for enumerated parameters
- Creating both **specific** and **unified** tool interfaces

## Prerequisites ✅

- Completed [Demo 01: Basic MCP Stdio](../demo-01-basic-mcp-stdio/)
- Understanding of Python functions and decorators
- Basic knowledge of type hints

## Quick Start 🚀

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run the Server

```bash
uv run python main.py
```

### 3. Test with MCP Inspector

In another terminal:

```bash
npx @modelcontextprotocol/inspector uv run python main.py
```

Visit http://localhost:5173 to interact with the calculator tools.

## Architecture Overview 🏗️

This demo implements a **calculator server** with 6 tools:

```
Calculator MCP Server
├── Basic Math Tools
│   ├── add(a, b)           → Addition
│   ├── subtract(a, b)      → Subtraction
│   ├── multiply(a, b)      → Multiplication
│   └── divide(a, b)        → Division (with zero check)
└── Advanced Math Tools
    ├── power(base, exp)    → Exponentiation
    └── modulo(a, b)        → Modulus operation
```

    └── get_calculator_info() → Server metadata

````

## Code Walkthrough 🔍

### 1. Multiple Tool Registration

FastMCP makes it easy to register multiple tools:

```python
mcp = FastMCP("Calculator Server")

@mcp.tool()
def add(a: float, b: float) -> dict:
    """Add two numbers"""
    result = a + b
    return {
        "result": result,
        "expression": f"{a} + {b} = {result}"
    }

@mcp.tool()
def subtract(a: float, b: float) -> dict:
    """Subtract b from a"""
    result = a - b
    return {
        "result": result,
        "expression": f"{a} - {b} = {result}"
    }
````

**Key Points:**

- Each `@mcp.tool()` decorator registers a new tool
- Function names become tool names
- Docstrings become tool descriptions
- Type hints define parameter schemas

### 2. Error Handling Pattern

The `divide` tool demonstrates proper error handling:

```python
@mcp.tool()
def divide(a: float, b: float) -> dict:
    """Divide a by b"""
    if b == 0:
        return {
            "error": "Division by zero is not allowed",
            "expression": f"{a} / {b} = undefined"
        }

    result = a / b
    return {
        "result": result,
        "expression": f"{a} / {b} = {result}"
    }
```

**Best Practices:**

- ✅ Check for invalid inputs before computation
- ✅ Return structured error messages
- ✅ Maintain consistent return format
- ✅ Provide helpful context (expression string)

### 3. Advanced Pattern: Unified Tool with Literal Types

**Note**: This is an optional advanced pattern not included in the demo. You can add it as an exercise!

A unified `calculate` tool shows advanced type usage with `Literal` for enums:

```python
from typing import Literal

@mcp.tool()
def calculate(
    operation: Literal["add", "subtract", "multiply", "divide", "power", "modulo"],
    a: float,
    b: float
) -> dict:
    """Unified calculator supporting multiple operations"""

    operations = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b if b != 0 else None,
        "power": lambda: a ** b,
        "modulo": lambda: a % b if b != 0 else None
    }

    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}

    result = operations[operation]()
    if result is None:
        return {"error": "Division/modulo by zero"}

    return {
        "result": result,
        "operation": operation,
        "expression": f"{a} {operation} {b} = {result}"
    }
```

**Why Literal Types?**

- 🎯 **Auto-completion**: MCP clients show available options
- ✅ **Validation**: Invalid operations rejected at protocol level
- 📝 **Documentation**: Clear API contract
- 🔍 **Type Safety**: IDE support and static analysis

### 4. Structured Return Values

All tools return consistent, structured data:

```python
{
    "result": 42.0,              # The computed value
    "expression": "40 + 2 = 42"  # Human-readable format
}

# Or for errors:
{
    "error": "Division by zero is not allowed",
    "expression": "10 / 0 = undefined"
}
```

**Benefits:**

- Easy to parse programmatically
- Clear for human readers
- Consistent error reporting
- Contextual information included

## Testing the Server 🧪

### Method 1: MCP Inspector (Recommended)

1. Start MCP Inspector:

   ```bash
   npx @modelcontextprotocol/inspector uv run python main.py
   ```

2. Open http://localhost:5173

3. Try these operations:

   **Basic Math:**

   ```json
   Tool: add
   { "a": 15, "b": 27 }
   ```

   **Division by Zero:**

   ```json
   Tool: divide
   { "a": 10, "b": 0 }
   ```

   **Unified Calculator:**

   ```json
   Tool: calculate
   { "operation": "power", "a": 2, "b": 10 }
   ```

### Method 2: Python MCP Client

Create a test script `test_calculator.py`:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_calculator():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "main.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test addition
            result = await session.call_tool("add", {"a": 10, "b": 5})
            print("Add:", result.content[0].text)

            # Test division by zero
            result = await session.call_tool("divide", {"a": 10, "b": 0})
            print("Divide by zero:", result.content[0].text)

            # Test unified calculator
            result = await session.call_tool("calculate", {
                "operation": "power",
                "a": 2,
                "b": 8
            })
            print("Power:", result.content[0].text)

asyncio.run(test_calculator())
```

Run it:

```bash
uv run python test_calculator.py
```

### Method 3: Manual stdio Testing

You can also test the raw JSON-RPC protocol:

```bash
uv run python main.py
```

Send (paste this JSON and press Enter):

```json
{ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }
```

Then call a tool:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": { "name": "add", "arguments": { "a": 10, "b": 5 } }
}
```

## Exercises 💪

### Exercise 1: Add Square Root Tool

Add a new tool for square root calculation:

```python
@mcp.tool()
def square_root(n: float) -> dict:
    """Calculate the square root of a number"""
    # Your implementation here
    pass
```

**Requirements:**

- Handle negative numbers (return error)
- Return both result and expression
- Include proper type hints

<details>
<summary>Solution</summary>

```python
import math

@mcp.tool()
def square_root(n: float) -> dict:
    """Calculate the square root of a number"""
    if n < 0:
        return {
            "error": "Cannot calculate square root of negative number",
            "expression": f"√{n} = undefined (complex)"
        }

    result = math.sqrt(n)
    return {
        "result": result,
        "expression": f"√{n} = {result}"
    }
```

</details>

### Exercise 2: Add Factorial Tool

Implement factorial calculation:

```python
@mcp.tool()
def factorial(n: int) -> dict:
    """Calculate the factorial of a non-negative integer"""
    # Your implementation here
    pass
```

**Requirements:**

- Only accept non-negative integers
- Handle large numbers (return error for n > 20)
- Return error for negative or decimal inputs

<details>
<summary>Solution</summary>

```python
import math

@mcp.tool()
def factorial(n: int) -> dict:
    """Calculate the factorial of a non-negative integer"""
    if n < 0:
        return {
            "error": "Factorial is not defined for negative numbers",
            "expression": f"{n}! = undefined"
        }

    if n > 20:
        return {
            "error": "Factorial too large (n > 20)",
            "expression": f"{n}! = too large to compute"
        }

    result = math.factorial(n)
    return {
        "result": result,
        "expression": f"{n}! = {result}"
    }
```

</details>

### Exercise 3: Add Percentage Tool

Create a tool to calculate percentages:

```python
@mcp.tool()
def percentage(value: float, percent: float) -> dict:
    """Calculate what percentage of a value is"""
    # Your implementation here
    pass
```

**Example:** `percentage(200, 15)` should return 30 (15% of 200)

### Exercise 4: Extend Unified Calculator

Add `square_root` and `factorial` to the `calculate` tool's operation list.

**Hint:** You'll need to modify the `Literal` type and the operations dictionary.

## Common Patterns 📋

### Pattern 1: Validation Before Computation

```python
@mcp.tool()
def safe_operation(value: float) -> dict:
    # Validate first
    if not is_valid(value):
        return {"error": "Validation failed"}

    # Then compute
    result = compute(value)
    return {"result": result}
```

### Pattern 2: Consistent Return Structure

```python
# Success
{"result": value, "expression": string}

# Error
{"error": message, "expression": string}
```

### Pattern 3: Typed Parameters

```python
# Specific types
def add(a: float, b: float) -> dict:
    ...

# Enumerated values
def calculate(op: Literal["add", "subtract"], a: float, b: float) -> dict:
    ...

# Optional parameters
from typing import Optional
def advanced_calc(a: float, b: float, precision: Optional[int] = 2) -> dict:
    ...
```

## Troubleshooting 🔧

### Issue: Division by zero doesn't return error

**Symptom:** Server crashes or returns infinity

**Solution:** Check for zero before division:

```python
if b == 0:
    return {"error": "Division by zero"}
```

### Issue: Literal type not validated

**Symptom:** Invalid operations accepted

**Solution:** Ensure you're using `Literal` from `typing`:

```python
from typing import Literal

def calculate(operation: Literal["add", "subtract"], ...):
    ...
```

### Issue: Tool not showing in MCP Inspector

**Symptom:** Some tools missing from tools list

**Solution:**

- Check for syntax errors in the tool function
- Ensure `@mcp.tool()` decorator is present
- Verify the function is defined before `mcp.run()`

### Issue: Type hints not working

**Symptom:** Parameters accept any type

**Solution:**

- Use proper Python type hints: `float`, `int`, `str`, `bool`
- Install types: `uv add --dev types-python`
- Check FastMCP version supports your type

## Key Takeaways 🎯

1. **Multiple Tools**: Use `@mcp.tool()` on multiple functions
2. **Validation**: Always validate inputs before computation
3. **Error Handling**: Return structured errors, don't raise exceptions
4. **Type Hints**: Use `Literal` for enums, proper types for validation
5. **Consistency**: Maintain consistent return structures
6. **Documentation**: Write clear docstrings for each tool

## What's Next? ⏭️

Continue to [Demo 03: Weather API Server](../demo-03-mcp-weather-server/) to learn:

- External API integration
- Environment configuration
- Async HTTP requests with httpx
- Demo mode for testing

## Resources 📚

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Error Handling Best Practices](https://docs.python.org/3/tutorial/errors.html)

## Need Help? 🆘

- Review [Demo 01](../demo-01-basic-mcp-stdio/) for basics
- Check the [MCP Documentation](https://modelcontextprotocol.io/)
- Join the [MCP Discord](https://discord.gg/modelcontextprotocol)

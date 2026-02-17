# Demo 03: MCP Weather API Server ☁️

## What You'll Learn 📚

In this demo, you'll learn:

- **External API integration** with OpenWeatherMap
- **Async HTTP requests** using httpx
- **Environment configuration** with python-dotenv
- **API key security** and best practices
- **Error handling** for network and API failures
- **Demo mode** for testing without API key

## Prerequisites ✅

- Completed [Demo 02: Calculator Tools](../demo-02-mcp-calculator-tools/)
- Free API key from [OpenWeatherMap](https://openweathermap.org/api)
- Understanding of async/await in Python

## Quick Start 🚀

### 1. Get API Key (Optional)

Sign up for a free OpenWeatherMap API key:
👉 https://openweathermap.org/api

**Note:** Demo works without API key (uses simulated data)

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env and add your API key:
# OPENWEATHER_API_KEY=your-actual-api-key-here
```

### 4. Run the Server

```bash
uv run python main.py
```

### 5. Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run python main.py
```

## Architecture Overview 🏗️

This demo implements a **Weather API Server** with 3 tools:

```
Weather MCP Server
├── get_weather(city, units)     → Current weather conditions
├── geocode(location)            → Location to coordinates
└── compare_weather(cities)      → Multi-city comparison
```

## Available Tools 📚

### 1. get_weather

```python
get_weather(city: str, units: str = "metric") -> dict
```

Get current weather conditions for a city.

**Parameters:**

- `city`: City name (e.g., "London", "New York, US")
- `units`: Temperature units - "metric" (Celsius), "imperial" (Fahrenheit), "kelvin"

**Returns:**

```json
{
  "city": "London",
  "temperature": 15.5,
  "feels_like": 14.2,
  "description": "partly cloudy",
  "humidity": 72,
  "wind_speed": 5.2,
  "units": "metric"
}
```

### 2. geocode

```python
geocode(location: str) -> dict
```

Convert location name to geographic coordinates.

**Parameters:**

- `location`: City name or address

**Returns:**

```json
{
  "name": "San Francisco",
  "latitude": 37.7749,
  "longitude": -122.4194,
  "country": "US"
}
```

### 3. compare_weather

```python
compare_weather(cities: list[str]) -> dict
```

Compare current weather across multiple cities.

**Parameters:**

- `cities`: List of city names

**Returns:**

```json
{
  "comparison": [
    { "city": "London", "temperature": 15, "description": "cloudy" },
    { "city": "Paris", "temperature": 18, "description": "sunny" },
    { "city": "Berlin", "temperature": 14, "description": "rainy" }
  ],
  "warmest": "Paris",
  "coldest": "Berlin"
}
```

## Key Concepts 🧠

### FastMCP with External APIs

FastMCP makes API integration simple:

```python
from fastmcp import FastMCP
import httpx

mcp = FastMCP("Weather Server")

@mcp.tool()
async def get_weather(city: str) -> dict:
    """Get current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url)
        data = response.json()
        return transform_data(data)
```

**Key Points:**

- Use `async def` for async I/O operations
- `httpx.AsyncClient()` for non-blocking HTTP
- Return structured dictionaries
- Handle errors gracefully with try/except

### API Key Security

**❌ Never do this:**

```python
API_KEY = "sk-1234567890"  # Hardcoded!
```

**✅ Always do this:**

```python
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("API key not found")
```

### Error Handling

Handle various API failures:

```python
try:
    response = await client.get(url)
    response.raise_for_status()  # Raises for 4xx/5xx
    return response.json()

except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        return {"error": "Invalid API key"}
    elif e.response.status_code == 404:
        return {"error": f"City not found: {city}"}
    elif e.response.status_code == 429:
        return {"error": "Rate limit exceeded"}
    else:
        return {"error": f"API error: {e}"}

except httpx.RequestError as e:
    return {"error": f"Network error: {e}"}
```

## 📁 Project Structure

```
demo-03-mcp-weather-server/
├── .python-version      # Python 3.12
├── .env                 # Your API key (git-ignored)
├── .env.example         # Template
├── .gitignore          # Protects .env
├── pyproject.toml      # Dependencies
├── README.md           # This file
└── main.py             # Weather server implementation
```

## 🔧 Troubleshooting

### Invalid API Key Error

```
ValueError: Invalid API key
```

**Solution:**

1. Check your `.env` file has the correct API key
2. Verify the key is active on OpenWeatherMap dashboard
3. Wait a few hours if you just created the key (activation delay)

### City Not Found

```
ValueError: City not found: Lodnon
```

**Solution:**

- Check spelling of city name
- Try format: "City, CountryCode" (e.g., "London, UK")
- Use geocode tool first to verify location

### Rate Limit Exceeded

```
ValueError: Rate limit exceeded
```

**Solution:**

- Free tier: 60 calls/minute, 1,000,000 calls/month
- Wait a minute before retrying
- Consider caching results
- Upgrade plan if needed

### Network Errors

```
ValueError: Network error: Connection timeout
```

**Solution:**

- Check your internet connection
- Verify OpenWeatherMap API is not down
- Check firewall/proxy settings

## 🎓 Learning Notes

### Why External APIs in MCP?

MCP excels at providing **real-time dynamic data**:

| Data Type            | Approach     | Use MCP?        |
| -------------------- | ------------ | --------------- |
| Real-time weather    | API call     | ✅ Yes          |
| Historical weather   | Database/RAG | ❌ No (use RAG) |
| Current stock prices | API call     | ✅ Yes          |
| Company financials   | Database/RAG | ❌ No (use RAG) |
| Live sports scores   | API call     | ✅ Yes          |
| Sports history       | Database/RAG | ❌ No (use RAG) |

### API Design Patterns

**Single Responsibility:**

```python
# ✅ Good: Each tool does one thing
@mcp.tool()
async def get_weather(city: str) -> dict: ...

@mcp.tool()
async def geocode(location: str) -> dict: ...
```

**Not:**

```python
# ❌ Bad: One tool tries to do everything
@mcp.tool()
async def weather(city: str, mode: str) -> dict:
    if mode == "current": ...
    elif mode == "geocode": ...
```

### Data Transformation

Transform API responses to clean, structured data:

```python
# Raw API response (complex)
raw = {
    "coord": {"lon": -0.1257, "lat": 51.5085},
    "weather": [{"id": 801, "main": "Clouds", "description": "few clouds"}],
    "main": {"temp": 288.15, "feels_like": 287.15, ...},
    ...
}

# Transformed (clean)
clean = {
    "city": "London",
    "temperature": 15.0,
    "description": "few clouds",
    "humidity": 72
}
```

## 💡 Exercise Ideas

Build these weather tools:

1. **get_air_quality** - Air quality index for a city
2. **get_uv_index** - UV radiation levels
3. **weather_alerts** - Get weather warnings/alerts
4. **historical_weather** - Weather from a specific past date

## 📚 Next Steps

1. **Demo 04** - Filesystem operations with path security
2. **Demo 05** - Database access tools
3. **Demo 08** - Use weather tools in an AI agent
4. **Demo 09** - Combine weather + other MCP servers

## 🔗 Resources

- [OpenWeatherMap API Docs](https://openweathermap.org/api)
- [httpx Documentation](https://www.python-httpx.org/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)

## 🤝 Need Help?

- Verify `.env` file exists and has your API key
- Test API key with curl: `curl "https://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_KEY"`
- Check OpenWeatherMap API status page
- Review error messages - they indicate the specific problem

---

**Happy Learning! 🚀**

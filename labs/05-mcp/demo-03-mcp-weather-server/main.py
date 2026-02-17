"""
Demo 03: MCP Weather API Server

This demo shows how to integrate external APIs into MCP tools:
- OpenWeatherMap API integration
- Async HTTP requests with httpx
- Environment variable configuration
- API error handling
- Data transformation

Key Concepts:
- External API integration
- Async I/O operations
- Secure credential management
- Real-world error scenarios
"""

import asyncio
import os
from typing import Optional, List
from dotenv import load_dotenv
from fastmcp import FastMCP
import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Environment variables
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = os.getenv("OPENWEATHER_BASE_URL", "https://api.openweathermap.org/data/2.5")
DEFAULT_UNITS = os.getenv("OPENWEATHER_UNITS", "metric")

# Validation
if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your-api-key-here":
    print("⚠️  WARNING: OPENWEATHER_API_KEY not configured")
    print("   This demo will run but API calls will be simulated.")
    print("   To use real weather data:")
    print("   1. Get free API key from https://openweathermap.org/api")
    print("   2. Copy .env.example to .env")
    print("   3. Add your API key to .env")
    print()
    DEMO_MODE = True
else:
    DEMO_MODE = False

# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP("Weather Server")

print("=" * 70)
print("MCP DEMO 03: WEATHER API SERVER")
print("=" * 70)
print(f"✓ API Mode: {'DEMO (simulated data)' if DEMO_MODE else 'LIVE (real API)'}")
print(f"✓ Base URL: {OPENWEATHER_BASE_URL}")
print(f"✓ Default Units: {DEFAULT_UNITS}")
print()

# ============================================================================
# WEATHER TOOLS
# ============================================================================


@mcp.tool()
async def get_weather(city: str, units: str = "metric") -> dict:
    """
    Get current weather conditions for a city.
    
    Args:
        city: City name (e.g., "London" or "New York, US")
        units: Temperature units - "metric" (Celsius), "imperial" (Fahrenheit), or "kelvin"
        
    Returns:
        Dictionary with weather information including temperature, description, humidity, wind speed
    """
    print(f"[Server] Tool 'get_weather' called: city={city}, units={units}")
    
    if DEMO_MODE:
        # Return simulated data for demo purposes
        return {
            "city": city,
            "temperature": 20.5,
            "feels_like": 19.8,
            "description": "partly cloudy",
            "humidity": 65,
            "wind_speed": 4.5,
            "units": units,
            "note": "DEMO MODE - simulated data"
        }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{OPENWEATHER_BASE_URL}/weather",
                params={
                    "q": city,
                    "appid": OPENWEATHER_API_KEY,
                    "units": units
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Transform API response to clean format
            result = {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "units": units
            }
            
            print(f"[Server] Weather data retrieved: {data['name']}, {result['temperature']}°")
            return result
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return {"error": "Invalid API key"}
        elif e.response.status_code == 404:
            return {"error": f"City not found: {city}"}
        elif e.response.status_code == 429:
            return {"error": "Rate limit exceeded - please wait"}
        else:
            return {"error": f"API error: {e.response.status_code}"}
            
    except httpx.RequestError as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def geocode(location: str) -> dict:
    """
    Convert location name to geographic coordinates.
    
    Args:
        location: City name or address
        
    Returns:
        Dictionary with name, latitude, longitude, and country
    """
    print(f"[Server] Tool 'geocode' called: location={location}")
    
    if DEMO_MODE:
        # Return simulated coordinates
        return {
            "name": location,
            "latitude": 51.5074,
            "longitude": -0.1278,
            "country": "GB",
            "note": "DEMO MODE - simulated data"
        }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{OPENWEATHER_BASE_URL.replace('/data/2.5', '/geo/1.0')}/direct",
                params={
                    "q": location,
                    "limit": 1,
                    "appid": OPENWEATHER_API_KEY
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return {"error": f"Location not found: {location}"}
            
            result = {
                "name": data[0]["name"],
                "latitude": data[0]["lat"],
                "longitude": data[0]["lon"],
                "country": data[0]["country"]
            }
            
            print(f"[Server] Coordinates found: {result['name']} ({result['latitude']}, {result['longitude']})")
            return result
            
    except httpx.HTTPStatusError as e:
        return {"error": f"Geocoding API error: {e.response.status_code}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {str(e)}"}
    except (KeyError, IndexError):
        return {"error": f"Location not found: {location}"}


@mcp.tool()
async def compare_weather(cities: List[str]) -> dict:
    """
    Compare current weather across multiple cities.
    
    Args:
        cities: List of city names (up to 5 cities)
        
    Returns:
        Dictionary with comparison data and warmest/coldest cities
    """
    print(f"[Server] Tool 'compare_weather' called: cities={cities}")
    
    if not cities or len(cities) == 0:
        return {"error": "No cities provided"}
    
    if len(cities) > 5:
        return {"error": "Maximum 5 cities allowed"}
    
    comparison = []
    
    for city in cities:
        try:
            weather = await get_weather(city)
            if "error" in weather:
                comparison.append({
                    "city": city,
                    "error": weather["error"]
                })
            else:
                comparison.append({
                    "city": weather["city"],
                    "temperature": weather["temperature"],
                    "description": weather["description"]
                })
        except Exception as e:
            comparison.append({
                "city": city,
                "error": str(e)
            })
    
    # Find warmest and coldest (excluding errors)
    valid_data = [c for c in comparison if "error" not in c]
    
    if not valid_data:
        return {"error": "No valid weather data retrieved"}
    
    warmest = max(valid_data, key=lambda x: x["temperature"])
    coldest = min(valid_data, key=lambda x: x["temperature"])
    
    result = {
        "comparison": comparison,
        "warmest": warmest["city"],
        "coldest": coldest["city"],
        "temperature_range": {
            "min": coldest["temperature"],
            "max": warmest["temperature"]
        }
    }
    
    print(f"[Server] Comparison complete: Warmest={warmest['city']}, Coldest={coldest['city']}")
    return result


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    mcp.run()

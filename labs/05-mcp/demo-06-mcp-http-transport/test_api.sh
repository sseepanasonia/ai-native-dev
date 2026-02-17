#!/bin/bash

echo "======================================================================"
echo "MCP HTTP Transport - API Tests"
echo "======================================================================"
echo ""

BASE_URL="http://localhost:8000"

echo "1. Health Check"
echo "----------------------------------------------------------------------"
curl -s $BASE_URL/ | jq '.'
echo ""

echo "2. List Tools"
echo "----------------------------------------------------------------------"
curl -s $BASE_URL/mcp/tools/list | jq '.'
echo ""

echo "3. Call Tool: greet"
echo "----------------------------------------------------------------------"
curl -s -X POST $BASE_URL/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "greet", "arguments": {"name": "Developer"}}' | jq '.'
echo ""

echo "4. Call Tool: add"
echo "----------------------------------------------------------------------"
curl -s -X POST $BASE_URL/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "add", "arguments": {"a": 15, "b": 27}}' | jq '.'
echo ""

echo "5. Call Tool with Error: divide by zero"
echo "----------------------------------------------------------------------"
curl -s -X POST $BASE_URL/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "divide", "arguments": {"a": 10, "b": 0}}' | jq '.'
echo ""

echo "======================================================================"
echo "Tests Complete!"
echo "======================================================================"

#!/bin/bash
# Start a local server for the LLM Math Visualiser UI
echo "🧠 LLM Math Visualiser"
echo "   Opening at: http://localhost:8080"
echo "   Press Ctrl+C to stop"
echo ""
cd "$(dirname "$0")"
python3 -m http.server 8080

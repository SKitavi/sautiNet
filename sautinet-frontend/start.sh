#!/bin/bash

echo "🇰🇪 Starting SautiNet Frontend Dashboard"
echo "========================================"
echo ""

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    echo "✓ Python 3 found"
    echo "Starting server on http://localhost:3000"
    echo ""
    python3 server.py
elif command -v python &> /dev/null; then
    echo "✓ Python found"
    echo "Starting server on http://localhost:3000"
    echo ""
    python -m http.server 3000
else
    echo "✗ Python not found"
    echo ""
    echo "Please install Python or use one of these alternatives:"
    echo "  - npx http-server -p 3000"
    echo "  - VS Code Live Server extension"
    exit 1
fi

#!/bin/bash
# Generate OpenAPI TypeScript client from FastAPI schema

set -e

echo "Generating OpenAPI client..."

# Check if backend is running
if ! curl -s http://localhost:17493/openapi.json > /dev/null 2>&1; then
    echo "Backend not running. Starting backend..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    
    # Install dependencies if needed
    if ! python -c "import fastapi" 2>/dev/null; then
        echo "Installing backend dependencies..."
        pip install -r requirements.txt
    fi
    
    # Start backend in background
    echo "Starting backend server..."
    uvicorn main:app --port 17493 &  # Keep the generator on the app's documented local backend port.
    BACKEND_PID=$!
    
    # Wait for server to be ready
    echo "Waiting for server to start..."
    for _ in {1..30}; do
        if curl -s http://localhost:17493/openapi.json > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:17493/openapi.json > /dev/null 2>&1; then
        echo "Error: Backend failed to start"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    
    echo "Backend started (PID: $BACKEND_PID)"
    STARTED_BACKEND=true
else
    STARTED_BACKEND=false
fi

# Download OpenAPI schema
echo "Downloading OpenAPI schema..."
curl -s http://localhost:17493/openapi.json > app/openapi.json

# Check if openapi-typescript-codegen is installed
if ! bunx --bun openapi-typescript-codegen --version > /dev/null 2>&1; then
    echo "Installing openapi-typescript-codegen..."
    bun add -d openapi-typescript-codegen
fi

# Generate TypeScript client
echo "Generating TypeScript client..."
cd app
bunx --bun openapi-typescript-codegen \
    --input openapi.json \
    --output src/lib/api \
    --client fetch \
    --useOptions \
    --exportSchemas true

echo "API client generated in app/src/lib/api"

# Clean up
if [ "$STARTED_BACKEND" = true ]; then
    echo "Stopping backend server..."
    kill $BACKEND_PID 2>/dev/null || true
fi

echo "Done!"

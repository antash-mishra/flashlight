#!/bin/bash

echo "Starting Tab Search Desktop App..."

# Activate virtual environment
source venv/bin/activate

# Install required packages if not already installed
pip install -q flask sentence-transformers scikit-learn requests faiss-cpu

# Test FAISS functionality
echo "Testing FAISS functionality..."
python test_faiss.py

# Start the desktop app
echo "Starting desktop app..."
python main.py &

# Wait a moment for the app to start
sleep 3

# Test the sync functionality
echo "Testing sync functionality..."
python test_sync.py

# Test memory persistence
echo "Testing memory persistence..."
python test_memory.py

echo "Desktop app is running on localhost:8080"
echo "Browser extension will automatically sync every 30 seconds"
echo "Memory is automatically saved and loaded"
echo "Press Ctrl+C to stop" 
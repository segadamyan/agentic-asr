#!/bin/bash

# Agentic ASR Frontend Setup Script

echo "🚀 Setting up Agentic ASR Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Node.js and npm are installed"

# Navigate to frontend directory
cd "$(dirname "$0")"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Ask user what they want to do
echo ""
echo "🎯 What would you like to do?"
echo "1) Start development server"
echo "2) Build for production"
echo "3) Exit"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🔥 Starting development server..."
        echo "Frontend will be available at http://localhost:3000"
        echo "Make sure the backend is running on http://localhost:8000"
        npm start
        ;;
    2)
        echo "🔨 Building for production..."
        npm run build
        if [ $? -eq 0 ]; then
            echo "✅ Production build completed successfully"
            echo "Build files are in the 'build' directory"
        else
            echo "❌ Production build failed"
            exit 1
        fi
        ;;
    3)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

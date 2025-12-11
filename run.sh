#!/bin/bash
# Financial Document Intelligence Platform - Run Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Financial Document Intelligence Platform${NC}"
echo -e "${GREEN}========================================${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "Python version: ${PYTHON_VERSION}"

# Function to start API server
start_api() {
    echo -e "\n${YELLOW}Starting API server on port 8000...${NC}"
    cd src/api
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    cd ../..
    echo -e "${GREEN}API server started (PID: $API_PID)${NC}"
    echo "API docs available at: http://localhost:8000/docs"
}

# Function to start Streamlit UI
start_ui() {
    echo -e "\n${YELLOW}Starting Streamlit UI on port 8501...${NC}"
    streamlit run ui/streamlit_app.py --server.port 8501 &
    UI_PID=$!
    echo -e "${GREEN}Streamlit UI started (PID: $UI_PID)${NC}"
    echo "UI available at: http://localhost:8501"
}

# Function to run tests
run_tests() {
    echo -e "\n${YELLOW}Running tests...${NC}"
    pytest tests/ -v
}

# Function to install dependencies
install_deps() {
    echo -e "\n${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
}

# Parse command line arguments
case "$1" in
    "api")
        start_api
        wait
        ;;
    "ui")
        start_ui
        wait
        ;;
    "all")
        start_api
        sleep 2
        start_ui
        echo -e "\n${GREEN}All services started!${NC}"
        echo "Press Ctrl+C to stop all services"
        wait
        ;;
    "test")
        run_tests
        ;;
    "install")
        install_deps
        ;;
    *)
        echo "Usage: $0 {api|ui|all|test|install}"
        echo ""
        echo "Commands:"
        echo "  api     - Start the FastAPI backend server"
        echo "  ui      - Start the Streamlit UI"
        echo "  all     - Start both API and UI"
        echo "  test    - Run the test suite"
        echo "  install - Install dependencies"
        exit 1
        ;;
esac

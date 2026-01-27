FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Expose ports for Streamlit and API
EXPOSE 8501
EXPOSE 8000

# Create a startup script
RUN echo '#!/bin/bash\n\
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & \n\
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]

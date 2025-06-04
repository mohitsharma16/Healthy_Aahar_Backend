# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "meal_planning_api:app", "--host", "0.0.0.0", "--port", "8000"]

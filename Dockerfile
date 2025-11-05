# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for API (optional, if using FastAPI)
EXPOSE 8000

# Default command to run training pipeline
# Change to prediction pipeline if needed
CMD ["python", "-m", "prediction.training_pipeline"]

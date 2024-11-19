# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set work directory
WORKDIR /code

# Copy entire project first
COPY . .

# Install venv package (required for virtual environment creation)
RUN apt-get update && apt-get install -y python3-venv && apt-get clean

# Create a virtual environment
RUN python -m venv .venv

# Activate the virtual environment and install dependencies
RUN . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/code/app
ENV PATH="/code/.venv/bin:$PATH"

# Command to run the application using the virtual environment's uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]

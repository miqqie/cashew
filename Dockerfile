# Stage 1: Build the image
FROM python:3.12-slim

# Install system dependencies needed for building Python packages (like pandas, numpy)
# The '-y' flag assumes yes to prompts, and 'rm -rf' cleans up cache to keep the image small.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files (Python scripts and CSV)
# **UPDATED to 'scripts/'**
COPY scripts/ /app/

# Create the .streamlit directory and copy the secrets.toml file
RUN mkdir -p .streamlit
COPY .streamlit/secrets.toml .streamlit/secrets.toml

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "dashboard_finalv.py", "--server.port", "8501", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
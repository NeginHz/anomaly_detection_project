# ---------- Dockerfile ----------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 8000

# Default command: start API
CMD ["python", "-m", "src.serve"]

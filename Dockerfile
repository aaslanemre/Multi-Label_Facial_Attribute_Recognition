# --- Dockerfile (Recommended for M-series) ---

# Use a standard Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# We install PyTorch here, which Docker will grab the ARM version of.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Set the default command
CMD ["python", "main.py"]
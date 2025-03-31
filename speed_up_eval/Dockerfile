# Use the specified Python Alpine base image
FROM python:3.11.0-slim

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATASET=emnist

# Install system dependencies required for PyTorch and building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# (Optional) If PyTorch requires additional dependencies, install them here
# For example, to install OpenBLAS for optimized linear algebra operations:
# RUN apk add --no-cache openblas-dev

# Copy the requirements.txt file into the container
COPY ./requirements.txt /app/requirements.txt

# Set the working directory
WORKDIR /app

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# (Optional) Copy your application code into the container
COPY . /app

# (Optional) Specify the command to run your application
CMD ["python", "src/main.py", "--epochs", "1", "--no-cuda", "--dataset", "emnist"]
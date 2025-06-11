# Use full Python image instead of slim to avoid missing dependencies
FROM python:3.10.12

WORKDIR /app

COPY . .

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ cmake git ninja-build libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Install scikit-build-core to fix build issues
RUN pip install --upgrade pip && \
    pip install --no-cache-dir scikit-build-core

# Install Python dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir --use-pep517 --no-build-isolation -r /app/requirements.txt

# Prevent pip from requiring hashes
ENV PIP_REQUIRE_HASHES=0

# Set the default command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10.12-slim

WORKDIR /app

COPY . .

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ cmake git libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Prevent pip from requiring hashes
ENV PIP_REQUIRE_HASHES=0

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir /app/hazm-0.10.0-py3-none-any.whl && \
    pip install --no-cache-dir --use-pep517 --no-build-isolation -r requirements.txt

# Set the default command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

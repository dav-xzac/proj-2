FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu &&\
    pip install --no-cache-dir \
        transformers==4.57.3 \
        datasets==4.4.2 \
        accelerate==1.12.0 \
        numpy==2.4.1 \
        pandas==2.3.3 
    
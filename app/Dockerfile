FROM ghcr.io/dav-xzac/mlops-base:latest AS builder

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
USER user

WORKDIR /app


COPY --from=builder /usr/local /usr/local

# Copia codice dell’app
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app_sentiment:app", "--host", "0.0.0.0", "--port", "7860"]


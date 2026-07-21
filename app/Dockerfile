FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-torch.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu 
RUN pip install --no-cache-dir -r requirements.txt
 
RUN useradd -m -u 1000 user
USER user

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app_sentiment:app", "--host", "0.0.0.0", "--port", "7860"]


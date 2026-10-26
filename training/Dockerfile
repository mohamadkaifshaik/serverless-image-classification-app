# Training image - only needed if you don't have a local Python with TensorFlow wheels.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ training/

ENTRYPOINT ["python", "training/train.py"]

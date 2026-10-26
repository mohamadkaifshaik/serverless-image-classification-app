FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080

WORKDIR /app

# deps first so the layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY model/ model/

# run as non-root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# 1 worker (TF itself is heavy in RAM), threads handle concurrent requests.
# --preload so the model loads once at boot rather than on the first request.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 --preload app.main:app

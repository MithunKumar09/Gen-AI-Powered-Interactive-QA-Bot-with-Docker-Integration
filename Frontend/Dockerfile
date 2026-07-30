FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Streamlit writes to ~/.streamlit at runtime; appuser owns its own home.
ENV HOME=/home/appuser

EXPOSE 8501

# Streamlit's own health endpoint. Probed with stdlib urllib because slim has
# no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8501')+'/_stcore/health', timeout=4).status==200 else sys.exit(1)"]

# Shell form so ${PORT} expands for Render. --server.address=0.0.0.0 is required
# for the container to be reachable at all.
CMD exec streamlit run app/app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true

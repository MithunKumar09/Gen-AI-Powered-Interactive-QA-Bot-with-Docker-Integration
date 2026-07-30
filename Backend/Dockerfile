# Python 3.12 is required, not preferred: pinecone >=6 and cohere >=7 both need
# >=3.10, and the previous base (python:3.9) is end-of-life. Slim keeps the image
# around 200MB instead of ~1GB.
FROM python:3.12-slim

# PYTHONUNBUFFERED is essential, not cosmetic: without it Python buffers stdout
# and structured logs never appear in `docker logs` or Render's log viewer until
# the buffer flushes.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so the layer is cached across source-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unprivileged. Combined with Backend/.dockerignore keeping .env out of the
# image, this closes the two container-level findings from the audit.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# python:*-slim ships no curl and no wget, so the usual `curl -f` healthcheck
# would fail with "not found" and report the container unhealthy forever.
# urllib is in the standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','5000')+'/health', timeout=4).status==200 else sys.exit(1)"]

# Shell form so ${PORT} expands -- Render injects PORT and the process must bind
# to it. Locally it defaults to 5000.
#
# Note the parentheses in "myapp:create_app()". Without them gunicorn treats the
# factory itself as the WSGI callable and invokes it as create_app(environ,
# start_response), which raises TypeError on every single request. The previously
# committed CMD omitted them.
#
# gthread with 2 workers x 4 threads: requests are I/O-bound on Cohere and
# Pinecone, so threads are the right lever. timeout 120 covers a slow embed of a
# large document; graceful-timeout gives in-flight requests a chance to finish on
# redeploy.
CMD exec gunicorn "myapp:create_app()" \
    --bind "0.0.0.0:${PORT:-5000}" \
    --worker-class gthread \
    --workers 2 \
    --threads 4 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile - \
    --log-level info

# Hugging Face Spaces — Docker SDK
# Streamlit app on free CPU-basic tier (16 GB RAM, 2 vCPU).

FROM python:3.11-slim

# HF Spaces best practice: run as non-root user (UID 1000).
# Without this, Streamlit's config dir (~/.streamlit) can hit permission errors.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR $HOME/app

# Install Python deps first so Docker layer-caches them across code changes.
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the app (src/, models/, prompts/, app.py, etc.)
COPY --chown=user . .

# HF Spaces routes external traffic to port 7860 by default for Docker SDK.
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]

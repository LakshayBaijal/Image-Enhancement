# Hugging Face Spaces (Docker SDK) or any machine with Docker. CPU only; 16 GB RAM is enough
# (measured peak 8.3 GB). Build: docker build -t product-enhance .   Run: docker run -p 8501:8501 -e HF_TOKEN=... product-enhance
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    ENHANCER_CACHE=/home/user/.cache/ondc-enhancer \
    ENHANCER_DEVICE=cpu \
    ENHANCER_THREADS=2 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
# XSRF/CORS off: Spaces embed the app in an iframe, where Streamlit's upload check otherwise fails with 403
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]

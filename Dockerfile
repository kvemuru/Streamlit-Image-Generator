FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "streamlit", "/bin/bash", "-c"]

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD conda run -n streamlit python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "streamlit", "streamlit", "run", "app_rev.py", "--server.address=0.0.0.0", "--server.port=8501"]

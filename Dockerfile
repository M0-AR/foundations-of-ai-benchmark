FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" \
    && python -m pip install -r requirements.txt \
    && python -m pip install "pytest>=8.0" "ruff>=0.6"

COPY . .
RUN python -m pip install --no-deps -e .

RUN mkdir -p results paper/figures data/raw/market

CMD ["python", "experiments/run_all.py"]

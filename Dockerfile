FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Instala dependências Python sem cache
RUN pip install -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ~/.cache/pip

# O Railway vai usar o startCommand do railway.json, então não precisa CMD aqui

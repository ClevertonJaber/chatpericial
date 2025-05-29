FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências de sistema necessárias
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

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Instala o modelo do spaCy diretamente aqui
RUN python -m spacy download pt_core_news_md

COPY . .

# Expõe a porta padrão para o Railway (opcional, mas recomendado)
EXPOSE 8080

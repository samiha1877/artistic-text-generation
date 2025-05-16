
FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py .
COPY bpe_tokenizer.json .
COPY poetry_rnn_model.pth .  


EXPOSE 8501


ENV STREAMLIT_PORT=8501
ENV STREAMLIT_BROWSER_SERVER_ADDRESS=0.0.0.0


CMD ["streamlit", "run", "app.py"]
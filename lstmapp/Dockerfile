
FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY app.py .
COPY word2idx.json .
COPY idx2word.json .
COPY poetry_generation_model.keras .


EXPOSE 8501


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
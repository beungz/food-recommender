FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level dependencies for xlearn, cmake, OpenCV, GTK, GThread
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY main.py ./main.py
COPY scripts/ ./scripts/
COPY models/deep_learning/ ./models/deep_learning/
COPY data/processed/ ./data/processed/
COPY data/outputs/ ./data/outputs/

EXPOSE 8080

CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]


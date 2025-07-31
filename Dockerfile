FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["bash"]

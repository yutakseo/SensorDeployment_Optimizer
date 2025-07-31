# 1. NVIDIA CUDA 12.2 베이스 (GPU 가능)
FROM nvidia/cuda:12.2.2-cudnn9-devel-ubuntu22.04

# 2. 작업 디렉토리 설정
WORKDIR /workspace

# 3. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. 로컬 코드 복사
COPY . /workspace

# 5. pip 업그레이드 및 requirements 설치
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 대기용 쉘
CMD ["bash"]

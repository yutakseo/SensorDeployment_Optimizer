# Python 3.12 slim 이미지를 기반으로 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치 (libGL.so.1 및 libglib2.0-0 포함)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# 로컬 파일을 컨테이너로 복사
COPY . /app

# pip 최신 버전으로 업그레이드 후 라이브러리 설치
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너 실행 시 수행할 명령어
CMD ["python", "Main.py"]

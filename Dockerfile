FROM python:3.8-slim  

# 기본 패키지 설치 (chromedriver 포함)
RUN apt-get update && \
    apt-get install -y wget unzip chromium-driver && \
    apt-get clean  

# Flask 앱과 필요한 파일 복사
COPY app.py /app/app.py  
COPY stopwordlist.txt /app/stopwordlist.txt  
COPY requirements.txt /app/requirements.txt  

WORKDIR /app  

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt  

# Flask 서버 실행
EXPOSE 5001  
CMD ["python", "app.py"]

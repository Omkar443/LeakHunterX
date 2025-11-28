FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install rich
COPY . .
CMD ["python3", "main.py"]

FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn openai requests pydantic nvidia-ml-py3
COPY main.py .
CMD ["uvicorn", "main:app", "--host","0.0.0.0","--port","8080"]
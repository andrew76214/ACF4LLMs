FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
RUN pip install fastapi uvicorn sympy
WORKDIR /app
COPY main.py .
CMD ["uvicorn", "main:app", "--host","0.0.0.0","--port","8020"]
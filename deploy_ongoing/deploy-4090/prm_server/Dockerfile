FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN pip install fastapi uvicorn transformers
COPY main.py .
ENV PRM_MODEL=/models/prm/final
CMD ["uvicorn", "main:app", "--host","0.0.0.0","--port","8010"]
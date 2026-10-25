FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md requirements.lock.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock.txt && pip install --no-cache-dir --no-deps -e .
COPY configs ./configs
COPY results/deployed ./results/deployed
COPY demo ./demo
EXPOSE 8000 8501
CMD ["uvicorn", "predictx.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .
COPY weather_plus /app/weather_plus
COPY .env.example /app/.env
EXPOSE 8000
CMD ["uvicorn", "weather_plus.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim-bullseye
ENV PYTHONUNBUFFERED 1

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for better visibility
EXPOSE 8000

# Use an entrypoint instead of a CMD. Entrypoint allows overriding with docker run
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "app.server:app"]
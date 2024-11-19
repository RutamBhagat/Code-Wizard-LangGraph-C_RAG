# Stage 1: Build
FROM python:3.11-slim-bullseye AS build
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Stage 2: Run
FROM python:3.11-slim-bullseye
WORKDIR /app
COPY --from=build /app/ .
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
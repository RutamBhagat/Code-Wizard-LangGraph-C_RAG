FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# Add extensive logging (This might not be as useful without venv, but keep it for now)
RUN pip show uvicorn > uvicorn_info.txt 2>&1

CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
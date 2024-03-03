FROM python:3.11.5-slim

# Install necessary system dependencies
RUN apt-get update \
    && apt-get install -y gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /Users/abhishekrajput/Desktop/orbit

COPY . /Users/abhishekrajput/Desktop/orbit

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV NAME orbit_env

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

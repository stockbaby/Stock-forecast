FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-docker.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ /app/
COPY test.sh /test.sh
COPY README.md /readme.md

RUN chmod +x /app/init.sh /app/train.sh /app/test.sh /test.sh

CMD ["sleep", "infinity"]

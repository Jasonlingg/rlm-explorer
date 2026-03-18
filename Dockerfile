FROM python:3.11-slim

RUN pip install --no-cache-dir pandas numpy regex tabulate

RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /workspace

CMD ["python3", "-i"]

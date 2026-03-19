FROM python:3.11-slim

RUN pip install --no-cache-dir pandas numpy regex tabulate

RUN useradd -m -s /bin/bash sandbox

# Bake corpus data into image so no volume mount is needed at runtime.
# Both synthetic (data/corpus) and MuSiQue (data/musique/corpus) are copied.
# Select active corpus at runtime via CORPUS_DIR env var.
COPY --chown=sandbox:sandbox data/ /workspace/data/

# Default corpus: synthetic. Override with -e CORPUS_DIR=/workspace/data/musique/corpus
ENV CORPUS_DIR=/workspace/data/corpus

USER sandbox
WORKDIR /workspace

CMD ["python3", "-i"]

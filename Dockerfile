FROM python:3.9-slim

RUN apt-get update && apt-get install -y gcc python3-dev

RUN pip install \
    sympy==1.12 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    gigachat==1.1.0 \
    langchain==0.2.2 \
    faiss-cpu==1.7.4

WORKDIR /app

CMD ["python", "--version"]
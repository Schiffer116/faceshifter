FROM python:3.10-slim

RUN apt-get update
RUN apt-get install -y git wget unzip zip
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workdir
RUN git clone https://github.com/Schiffer116/faceshifter.git
RUN mv faceshifter/* .
RUN rm -rf faceshifter

WORKDIR /data
RUN curl -L -o ffhq-256x256.zip \
    https://www.kaggle.com/api/v1/datasets/download/rahulbhalley/ffhq-256x256
RUN curl -L -o celebahq-resized-256x256.zip \
    https://www.kaggle.com/api/v1/datasets/download/badasstechie/celebahq-resized-256x256
RUN unzip ffhq-256x256.zip
RUN unzip celebahq-resized-256x256.zip

WORKDIR /workdir/preprocess
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir output
RUN python preprocess.py --root /data --output output

CMD zip -r /workdir/preprocess.zip /workdir/preprocess/output

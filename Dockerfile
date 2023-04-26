FROM python:3.11-slim-buster

MAINTAINER Flyte Team <users@flyte.org>
LABEL org.opencontainers.image.source https://github.com/flyteorg/flytekit

WORKDIR /root
ENV PYTHONPATH /root

ARG VERSION

RUN apt-get update && apt-get install build-essential -y

RUN pip install -U flytekit==$VERSION \
    flytekitplugins-pod==$VERSION \
    flytekitplugins-deck-standard==$VERSION \
    flytekitplugins-polars==$VERSION \
    flytekitplugins-papermill==$VERSION \
    scikit-learn

COPY requirements.txt /root
RUN pip install -r /root/requirements.txt

RUN useradd -u 1000 flytekit
RUN chown flytekit: /root
USER flytekit

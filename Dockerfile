FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

USER root

WORKDIR /app

# RUN apt update

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY .env .env

RUN wandb login $(cat .env/wandb)

# COPY . . # copy해서 image를 만드는 대신 그냥 workdir mount로 하는게 훨씬 나음


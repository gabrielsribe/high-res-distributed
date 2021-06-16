FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
COPY ./requirements.txt /root/requirements.txt
WORKDIR /root
RUN pip install -r requirements.txt

FROM ufoym/deepo:pytorch-py36
COPY ./requirements.txt /root/requirements.txt
WORKDIR /root
RUN pip install -r requirements.txt
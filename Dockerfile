FROM tensorflow/tensorflow:latest-py3

# Install dependencies
RUN pip3 install keras

RUN mkdir -p /client

# Add files to /client in docker container
COPY ./main.py /client
COPY ./senti_train.py /client
COPY ./fl_round /client/fl_round

WORKDIR /client

ENTRYPOINT [ "python3", "main.py"]
FROM tensorflow/tensorflow:latest-py3

# Install dependencies
RUN pip3 install keras

# Add this directory to /client in docker container
ADD . /client

RUN mkdir -p /data/weight_updates

WORKDIR /client

ENTRYPOINT [ "python3", "main.py"]
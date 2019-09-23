FROM tensorflow/tensorflow:latest-py3

# Install dependencies
RUN pip3 install keras numpy

# Add this directory to /client in docker container
ADD . /client

RUN mkdir /dataweight_updates

WORKDIR /client

ENTRYPOINT [ "python3", "main.py", "../data" ]
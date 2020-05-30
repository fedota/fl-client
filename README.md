# Client
Python Client for the Federated learning system

## Overview
Client software is used by data organization providing data for a particular FL problem and has the following responsibilities.
- Connect with the Selector to indicate availability to participate in the FL round
- Train the received model on the data available locally
- Report new checkpoint and weight (amount of data used) back to the Selector

## Workflow
- Each client for specific fl problem knows the address of the Selector to contact and sends a connection request to express availability
- After it gets selected it receives the initial files like model, checkpoint, etc. for the problem through the Selector and starts training the model with the local data
- Once training is complete the updated checkpoint and weight, no of training batches, are sent to the Selector 

## Setup

### Run with docker

* Build the client docker image:
`docker build -t fl-client .`

* Spawn a container with device specific data mounted:
`docker run -it --network="host" -v /path_to_device_data/device1:/device_data -v /path_to/config.yaml:/client/config.yaml --name client1 fl-client` 
    
    For example -  `docker run -it --network="host" -v $PWD/../device1:/device_data -v $PWD/../device1/config/config.yaml:/client/config.yaml --name client1 fl-client`

* To restart the container: 
`docker start -i client1`

* To inspect the running container, open bash using:
`docker exec -t -i client1 /bin/bash`

* To remove the container:
`docker rm client1`

* To simply run and inspect a new container, execute:
`docker run -it fl-client bash`


### Run without docker

* Create a virtual environment with
`virtualenv fl`

* Activate virtual environment with
`source fl/bin/activate`
 
* Install dependencies using
`pip install -r requirements.txt`

* Make necessary changes in `config.yaml`

* Run fl-client with
`python main.py`
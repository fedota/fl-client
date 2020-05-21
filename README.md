# fl-client
Python Client for simulating a device for federated learning

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
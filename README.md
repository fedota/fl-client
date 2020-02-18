# fl-client
Python Client for simulating a device for federated learning

* Build the client docker image
`docker build -t fl-client .`

* Spawn a container with device specific data mounted
`docker run -it --network="host" -v /path_to_device_data/device1:/data -e FL_DATASET_ID=1 fl-client`

FL_DATASET_ID is the environment variable used for using a subset of the dataset for simulating a particular federated round of training on the client
It can range from [1,5]

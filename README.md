# fl-client
Python Client for simulating a device for federated learning

* Build the client docker image
`docker build -t fedota-client .`

* Spawn a container with device specific data mounted
`docker run -rm -it -v $DEVICE_DATA_PATH:/data fedota-client`

docker run -it --network="host" -v /media/pvgupta24/MyZone/Projects/go/src/federated-learning/device_data/device1:/data fedota-client
Check:
OSError: Unable to create file (unable to open file: name = '../dataweight_updates/fl_weight_updates', errno = 2, error message = 'No such file or directory', flags = 13, o_flags = 242)

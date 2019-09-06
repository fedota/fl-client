
import logging

from google.protobuf import empty_pb2
import grpc

from fl_round import fl_round_pb2
from fl_round import fl_round_pb2_grpc

from train import train_on_device

_TIMEOUT_SECONDS = 10000
address = 'localhost:50051'
checkpoint_file_path = 'data/client/checkpoint.txt'
updated_checkpoint_file_path = 'data/client/updated_checkpoint.txt'
model_path = '../fl-misc/data/model.h5'
device_path = 'data/'
_CHUNKER_SIZE = 5


def checkInMessages():
    msg = fl_round_pb2.CheckInRequest(message='PythonClient')
    yield msg


def get_updates(updated_checkpoint_file_path, num_batches, chunker_size):
    with open(updated_checkpoint_file_path, "rb") as file:
        chunk = file.read(chunker_size)
        while chunk:
            print(str(chunk))
            yield fl_round_pb2.FlData(
                message=fl_round_pb2.Chunk(content=chunk),
                type=fl_round_pb2.FL_CHECKPOINT_UPDATE,
                intVal=num_batches
            )
            chunk = file.read(chunker_size)


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(address) as channel:
        client = fl_round_pb2_grpc.FlRoundStub(channel)

        with open(checkpoint_file_path, 'ab') as checkpoint_file:
            try:
                # responses = client.CheckIn(empty_pb2.Empty())
                responses = client.CheckIn(checkInMessages(), _TIMEOUT_SECONDS)

                # responses.send(fl_round_pb2.CheckInRequest(message = 'TestClientPython'))
                print(responses)
                # received_bytes = bytes()
                for response in responses:
                    print(response)
                    checkpoint_file.write(response.message.content)
                    # print('Received %d bytes...', len(response.chunk))

                print('Checkpoint file downloaded successfully')

                # Call helper functions from train.py to train and send back
                # TODO: Train
                num_batches, updated_weights_path = train_on_device(device_path, model_path, checkpoint_file_path)
                updated_chunks = get_updates(
                    updated_weights_path, num_batches, _CHUNKER_SIZE)
                    
                # print(updates_chunks[0])
                response = client.Update(updated_chunks, _TIMEOUT_SECONDS)

            except grpc.RpcError as rpc_error:
                print('Encountered a RPC error with ' + address)
                raise rpc_error


if __name__ == '__main__':
    logging.basicConfig()
    run()

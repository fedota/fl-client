
import logging
import sys

from google.protobuf import empty_pb2
import grpc

from fl_round import fl_round_pb2
from fl_round import fl_round_pb2_grpc

from train import train_on_device

_TIMEOUT_SECONDS = 10000
address = 'localhost:50051'


_CHUNKER_SIZE = 200


def checkInMessages():
    msg = fl_round_pb2.CheckInRequest(message='PythonClient')
    yield msg

# def get_updates_metadata(weight_updates_file_path, num_batches, chunker_size):
#     with open(weight_updates_file_path, "rb") as file:
#         chunk = file.read(chunker_size)
#         while chunk:
#             #print(str(chunk))
#             yield fl_round_pb2.FlData(
#                 message=fl_round_pb2.Chunk(content=chunk),
#                 type=fl_round_pb2.FL_CHECKPOINT_UPDATE,
#                 intVal=num_batches
#             )
#             chunk = file.read(chunker_size)

def get_weight_updates(weight_updates_file_path, num_batches, chunker_size):
    with open(weight_updates_file_path, "rb") as file:
        chunk = file.read(chunker_size)
        while chunk:
            #print(str(chunk))
            yield fl_round_pb2.FlData(
                message=fl_round_pb2.Chunk(content=chunk),
                type=fl_round_pb2.FL_CHECKPOINT_UPDATE,
                #intVal=num_batches
            )
            chunk = file.read(chunker_size)

    yield fl_round_pb2.FlData(
        type=fl_round_pb2.FL_CHECKPOINT_WEIGHT,
        intVal=num_batches
    )

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel(address) as channel:
        client = fl_round_pb2_grpc.FlRoundStub(channel)

        with open(checkpoint_file_path, 'wb') as checkpoint_file:
            try:
                # responses = client.CheckIn(empty_pb2.Empty())
                responses = client.CheckIn(checkInMessages(), _TIMEOUT_SECONDS)

                # responses.send(fl_round_pb2.CheckInRequest(message = 'TestClientPython'))
                #print(responses)
                # received_bytes = bytes()
                for response in responses:
                    #print(response)
                    checkpoint_file.write(response.message.content)
                    # print('Received %d bytes...', len(response.chunk))

                print('------- Checkpoint file downloaded successfully ------')

                

            except grpc.RpcError as rpc_error:
                print('Encountered a RPC error with ' + address)
                raise rpc_error
            
        # Call helper functions from train.py to train and send back
        # TODO: Train
        num_batches, weight_updates_path = train_on_device(data_dir, model_file_path, checkpoint_file_path, weight_updates_file_path)
        
        print('----- Completed training on device -----')

        updated_chunks = get_weight_updates(
            weight_updates_path, num_batches, _CHUNKER_SIZE)        
        #print(updated_chunks)
        response = client.Update(updated_chunks, _TIMEOUT_SECONDS)
        print('----- Weight updates sent to server -----')

if __name__ == '__main__':

    if len (sys.argv) != 2 :
        print("Usage: python main.py <path to directory containing device's content>")
        sys.exit (1)

    device_dir = sys.argv[1]
    
    data_dir = device_dir + '/data/'
    checkpoint_file_path = device_dir + '/checkpoint/fl_checkpoint'
    model_file_path = device_dir + '/model/model.h5'
    weight_updates_file_path = device_dir + 'weight_updates/fl_weight_updates'

    logging.basicConfig()
    run()

"""Simulation for clients in federated-learning."""
import logging
import os

import grpc
import yaml

from fl_round import fl_round_pb2
from fl_round import fl_round_pb2_grpc
from train import train_on_device

CONFIG_FILE = "config.yaml"
config = {}


def get_save_path(fileName):
    """Directory path to store required files."""
    return os.path.join(config["device-dir"], fileName)


def checkInMessages():
    """Checkin request to be sent to selectors."""
    msg = fl_round_pb2.CheckInRequest(message="PythonClient")
    yield msg


def get_weight_updates(weight_updates_file_path, num_batches, chunker_size):
    """Gets the checkpoint weights."""
    with open(weight_updates_file_path, "rb") as file:
        chunk = file.read(chunker_size)
        while chunk:
            # print(str(chunk))
            yield fl_round_pb2.FlData(
                message=fl_round_pb2.Chunk(content=chunk),
                type=fl_round_pb2.FL_CHECKPOINT_UPDATE,
                # intVal=num_batches
            )
            chunk = file.read(chunker_size)

    yield fl_round_pb2.FlData(
        type=fl_round_pb2.FL_CHECKPOINT_WEIGHT, intVal=num_batches,
    )


def run():
    """Starts the grpc server to connect to selectors."""
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    address = config["address"]

    with grpc.insecure_channel(address) as channel:
        client = fl_round_pb2_grpc.FlRoundStub(channel)

        try:
            responses = client.CheckIn(checkInMessages(), config["timeout-in-seconds"])

            for response in responses:
                if response.type == fl_round_pb2.FL_INT:
                    print(
                        "Could not Check in. Reconnect after "
                        + str(response.intVal)
                        + " seconds",
                    )
                    # TODO: Reconnect after some time
                    exit(-1)

                elif response.type == fl_round_pb2.FL_FILES:
                    with open(
                        get_save_path(response.filePath), "a+",
                    ) as checkpoint_file:
                        checkpoint_file.write(response.message.content)

            print("------- Checkpoint file downloaded successfully ------")

        except grpc.RpcError as rpc_error:
            print("Encountered a RPC error with " + address)
            raise rpc_error

        num_batches, weight_updates_path = train_on_device(
            data_dir,
            dataset_id,
            model_file_path,
            checkpoint_file_path,
            weight_updates_file_path,
        )

        print("----- Completed training on device -----")

        updated_chunks = get_weight_updates(
            weight_updates_path, num_batches, config["chunker-size"],
        )
        # print(updated_chunks)
        response = client.Update(updated_chunks, config["timeout-in-seconds"])
        print("----- Weight updates sent to server -----")


if __name__ == "__main__":

    config = yaml.load(open(CONFIG_FILE, "r"))

    device_dir = config["device-dir"]
    dataset_id = config["dataset-id"]

    data_dir = device_dir + "/data/"
    checkpoint_file_path = device_dir + "/checkpoint/fl_checkpoint"
    model_file_path = device_dir + "/model/model.h5"
    weight_updates_file_path = device_dir + "/weight_updates/fl_weight_updates"

    logging.basicConfig()
    run()

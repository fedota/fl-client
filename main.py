"""Simulation for clients in federated-learning."""
import logging
import os
import shutil

import grpc
import yaml

from fl_round import fl_round_pb2
from fl_round import fl_round_pb2_grpc
from senti_train import train_on_device

CONFIG_FILE = "config.yaml"
config = {}


def getSavePath(fileName):
    """Directory path to store required files."""
    return os.path.join(config["DEVICE_DIR"], config["FL_INIT_DIR"], fileName)

def createDirectories():
    fl_init_files_dir = os.path.join(config["DEVICE_DIR"], config["FL_INIT_DIR"])
    # Delete fl_init_files dir if it exists
    if os.path.exists(fl_init_files_dir):
        shutil.rmtree(fl_init_files_dir)
    os.makedirs(fl_init_files_dir)          # Create empty fl_init_files dir

    weight_updates_dir = os.path.join(config["DEVICE_DIR"], config["WEIGHT_UPDATES_DIR"])
    # Delete weight_updates dir if it exists
    if os.path.exists(weight_updates_dir):
        shutil.rmtree(weight_updates_dir)
    os.makedirs(weight_updates_dir)          # Create empty weight_updates dir



def checkInMessages():
    """Checkin request to be sent to selectors."""
    msg = fl_round_pb2.CheckInRequest(message="PythonClient")
    yield msg


def getWeightUpdates(weight_updates_file_path, num_batches, chunker_size):
    """Gets the checkpoint weights."""
    with open(weight_updates_file_path, "rb") as file:
        chunk = file.read(chunker_size)
        while chunk:
            yield fl_round_pb2.FlData(
                chunk=chunk,
                type=fl_round_pb2.FL_FILES,
            )
            chunk = file.read(chunker_size)

    yield fl_round_pb2.FlData(
        type=fl_round_pb2.FL_INT, 
        intVal=num_batches,
    )

def run():
    """Starts the grpc server to connect to selectors."""
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    address = config["SELECTOR_ADDRESS"]

    with grpc.insecure_channel(address) as channel:
        client = fl_round_pb2_grpc.FlRoundStub(channel)

        try:
            responses = client.CheckIn(checkInMessages(), config["TIMEOUT_IN_SECONDS"])

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
                        getSavePath(response.filePath), "ab+",
                    ) as CHECKPOINT_FILE:
                        CHECKPOINT_FILE.write(response.chunk)

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

        updated_chunks = getWeightUpdates(
            weight_updates_path, num_batches, config["CHUNKER_SIZE"],
        )
        # print(updated_chunks)
        response = client.Update(updated_chunks, config["TIMEOUT_IN_SECONDS"])
        print("----- Weight updates sent to server -----")


if __name__ == "__main__":

    config = yaml.load(open(CONFIG_FILE, "r"))

    data_dir = os.path.join(config["DEVICE_DIR"], config["DATA_DIR"])
    checkpoint_file_path = os.path.join(config["DEVICE_DIR"], config["FL_INIT_DIR"], config["CHECKPOINT_FILE"])
    model_file_path = os.path.join(config["DEVICE_DIR"], config["FL_INIT_DIR"], config["MODEL_FILE"])
    weight_updates_file_path = os.path.join(config["DEVICE_DIR"], config["WEIGHT_UPDATES_DIR"], config["WEIGHT_UPDATES_FILE"])
    dataset_id = config["DATASET_ID"]

    logging.basicConfig()

    createDirectories()

    run()

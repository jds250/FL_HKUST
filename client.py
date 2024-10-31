import argparse
import logging
import os

import flwr as fl
import torch

import FedoSSL.models as models
from Fedcustom import set_parameters,get_parameters

import utils
from helpers.load_data import load_datasets

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--batch_size", type=int, default=1024, help="Batch size for training"
)
# parser.add_argument(
#     "--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer"
# )
parser.add_argument("--client_id", type=int, default=1, help="Unique ID for the client")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total number of clients"
)
parser.add_argument(
    "--data_percentage", type=float, default=0.5, help="Portion of client data to use"
)

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.client_id = args.client_id

        # print(self.client_id)

        logger.info("Preparing data...")

        self.net = models.resnet18(num_classes=10).to(device)

        self.name_filters = ["mem_projections", "local_centroids", "local_labeled_centroids"]

        self.filtered_params = {}

        client_train_label_loader,client_train_unlabel_loader,client_test_loader = load_datasets()

        self.client_train_label_loader = client_train_label_loader[self.client_id]
        self.client_train_unlabel_loader = client_train_unlabel_loader[self.client_id]
        self.client_test_loader = client_test_loader[self.client_id]

    def extract_filtered_parameters(self):
        # 用于存储符合条件的参数
        for name, param in self.net.named_parameters():
            # 如果参数名称包含在 name_filters 中，则提取该参数
            if any(filter_name in name for filter_name in self.name_filters):
                self.filtered_params[name] = param

    def update_filtered_parameters(self):
        # 将 new_params 中的参数覆盖到模型的对应层
        for name, param in self.net.named_parameters():
            if name in self.filtered_params:
                param.data = self.filtered_params[name].data.clone()

    def get_parameters(self, config):
        # Return the parameters of the model
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Set the weights of the model
        set_parameters(self.net,parameters)
        epochs = 5
        labeled_num = 5
        global_round = 40

        self.update_filtered_parameters()
        # Train the model
        for epoch in range(epochs):
            mean_uncert,_ = utils.test(args, self.net, labeled_num, device, self.client_test_loader, epoch, self.client_id)
            utils.train(args, self.net, device, self.client_train_label_loader, self.client_train_unlabel_loader, mean_uncert,global_round)
        # local_clustering #
        self.net.local_clustering(device=device)
        # Directly return the parameters and the number of examples trained on
        self.extract_filtered_parameters()

        return get_parameters(self.net), len(self.client_train_label_loader), {}

    def evaluate(self, parameters, config):
        # Set the weights of the model
        set_parameters(self.net,parameters)
        labeled_num = 5
        # Evaluate the model and get the loss and accuracy
        loss, accuracy = utils.test(args, self.net, labeled_num, device, self.client_test_loader, 0, self.client_id, is_print=False)

        # Return the loss, the number of examples evaluated on and the accuracy
        return float(loss), len(self.client_test_loader), {"accuracy": float(accuracy)}

# Function to Start the Client
def start_fl_client():
    try:
        client = Client(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()

